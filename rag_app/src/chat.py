"""ConversationalRAG: orchestrates retrieval + grounded generation + history.

This is the single-document conversational path. intelligence.py implements the
multi-document and hierarchical variant. The two share the retrieval primitives
but route differently, so behaviour changes here need a deliberate decision
about whether the other path should follow — including the refusal thresholds
below, which currently differ between the two.

One turn:
  question -> decompose/rewrite -> fiscal expansion -> retrieve (+ rerank)
           -> refusal check -> generate -> append to history

Why the refusal check runs before the LLM call, not after:
  If the best candidate is weak on BOTH the dense and the sparse signal, the
  evidence is not in the index and no prompt will conjure it. Refusing here
  saves a round trip, and more importantly removes the opportunity for the model
  to fall back on parametric knowledge and answer from memory instead of from
  the document.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from . import llm, rerank
from .retriever import Hit, HybridRetriever


# Fiscal-period expansion (Indian fiscal calendar: Apr → Mar)
# Indian financial reports use date-ended terminology ("Quarter ended 30-09-2025")
# while users naturally ask in shorthand ("Q2 FY26"). This expander appends the
# date-ended phrasing to the RETRIEVAL query so BM25 + dense both find the right
# chunks. The LLM still sees the user's original question — we only widen recall.
#
# Indian FY mapping:
#   FY26 = April 2025 → March 2026  (year ending March 31)
#   Q1 FY26 = Apr-Jun 2025 → quarter ended 30-06-2025
#   Q2 FY26 = Jul-Sep 2025 → quarter ended 30-09-2025
#   Q3 FY26 = Oct-Dec 2025 → quarter ended 31-12-2025
#   Q4 FY26 = Jan-Mar 2026 → quarter ended 31-03-2026
#   H1 FY26 = Apr-Sep 2025 → half year ended 30-09-2025
#   H2 FY26 = Oct-Mar 2026 → half year ended 31-03-2026

# Match Q[1-4] FYxx, Q[1-4]-xx, Q[1-4] xx, Q[1-4]FYxx (and H[12] variants).
_QFY_RE = re.compile(r"\b(Q[1-4])[\s\-/]*(?:FY)?[\s\-/]*(\d{2,4})\b", re.IGNORECASE)
_HFY_RE = re.compile(r"\b(H[12])[\s\-/]*(?:FY)?[\s\-/]*(\d{2,4})\b", re.IGNORECASE)
_FY_RE  = re.compile(r"\bFY[\s\-/]*(\d{2,4})\b", re.IGNORECASE)

_Q_END = {"Q1": ("30-06", -1), "Q2": ("30-09", -1), "Q3": ("31-12", -1), "Q4": ("31-03", 0)}


def _normalize_fy(yy_str: str) -> int:
    """FY26 / FY2026 → 2026 (year ending March 31)."""
    n = int(re.sub(r"[^\d]", "", yy_str))
    return n + 2000 if n < 100 else n


def expand_fiscal_terms(q: str) -> str:
    """Append date-ended phrasings for Indian fiscal-period shorthand. The
    user-visible question is unchanged; this only enriches the retrieval query."""
    extras: list[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        """Append a phrasing once, preserving order."""
        if s not in seen:
            seen.add(s); extras.append(s)

    for m in _QFY_RE.finditer(q):
        q_label = m.group(1).upper()
        fy_yr = _normalize_fy(m.group(2))
        end_md, year_off = _Q_END[q_label]
        end_yr = fy_yr + year_off
        add(f"quarter ended {end_md}-{end_yr}")
        # also a few common date wordings (so dense embeddings catch them)
        month = {"30-06": "30 June", "30-09": "30 September",
                 "31-12": "31 December", "31-03": "31 March"}[end_md]
        add(f"{month} {end_yr}")

    for m in _HFY_RE.finditer(q):
        h_label = m.group(1).upper()
        fy_yr = _normalize_fy(m.group(2))
        if h_label == "H1":
            add(f"half year ended 30-09-{fy_yr - 1}")
            add(f"30 September {fy_yr - 1}")
        else:
            add(f"half year ended 31-03-{fy_yr}")
            add(f"31 March {fy_yr}")

    # Bare "FY26" with no Q/H prefix → financial year ending date
    if not extras:
        for m in _FY_RE.finditer(q):
            fy_yr = _normalize_fy(m.group(1))
            add(f"year ended 31-03-{fy_yr}")
            add(f"31 March {fy_yr}")

    if not extras:
        return q
    return q + " " + " ".join(extras)


# Pull a wider candidate set when reranking, then trim.
# Bigger candidate pool → reranker has more options → broader recall on
# open-ended questions ("summarize", "list all", "what was approved").
RERANK_CANDIDATE_K = 30


# Open-ended question detector. When matched, we widen top_k so the LLM gets
# enough chunks to enumerate every relevant item per the COMPLETENESS rule.
_BROAD_QUESTION_RE = re.compile(
    r"\b(summari[sz]e|list (?:all|every|the)|what (?:are|were) all|"
    r"all (?:of |the )?(?:items?|points?|approvals?|highlights?|drivers?|segments?)|"
    r"what (?:did|does|was) .* (?:approve|cover|include|discuss|mention)|"
    r"give me (?:all|every)|enumerate|outline|overview)\b",
    re.IGNORECASE,
)
BROAD_TOP_K = 12  # widened recall when the question is broad
DEFAULT_TOP_K = 8


# Empirical floors for refusal. We use the higher of dense cosine and a
# normalized BM25 signal. Tuned so generic chit-chat / off-doc Qs refuse early.
MIN_DENSE_SCORE = 0.45
MIN_SPARSE_SCORE = 1.5


@dataclass
class Turn:
    """One question/answer exchange, with the evidence and token cost behind it."""
    question: str
    rewritten: str
    answer: str
    hits: List[Hit] = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    refused: bool = False


def format_context(hits: List[Hit], max_chars_per_chunk: int = 1800) -> str:
    """Render hits into the CONTEXT block the prompt consumes.

    Each chunk is preceded by its citation so the model can attribute claims,
    and truncated so one oversized table cannot consume the whole budget.
    """
    parts = []
    for h in hits:
        snippet = h.chunk.text
        if len(snippet) > max_chars_per_chunk:
            snippet = snippet[:max_chars_per_chunk].rstrip() + " …"
        parts.append(f"{h.chunk.citation}\n{snippet}")
    return "\n\n---\n\n".join(parts)


def _effective_top_k(question: str, base: int) -> int:
    """Widen recall for broad questions ("summarize", "list all", etc.)."""
    if _BROAD_QUESTION_RE.search(question):
        return max(base, BROAD_TOP_K)
    return base


class ConversationalRAG:
    """Single-document conversational RAG with history.

    History lives in memory for the life of the object, so it is per-session,
    not persisted. The server holds one instance per conversation.
    """
    def __init__(self, retriever: HybridRetriever, top_k: int = DEFAULT_TOP_K):
        """Bind a retriever and set the default recall width."""
        self.retriever = retriever
        self.top_k = top_k
        self.history: List[dict] = []

    def reset(self) -> None:
        """Clear history, starting a fresh conversation on the same document."""
        self.history.clear()

    def ask(self, question: str) -> Turn:
        """Answer one question and record the turn.

        Rewrites the question into standalone sub-queries, expands fiscal shorthand
        for retrieval only, retrieves and optionally reranks per sub-query, then
        merges by citation keeping the best-scoring copy of each chunk.

        Refuses before calling the LLM when nothing retrieved clears both score
        floors. The post-hoc check on the answer text is a second net for cases
        where retrieval looked adequate but the model still found no answer.
        """
        sub_queries = llm.decompose_and_rewrite(self.history, question)
        rewritten = " | ".join(sub_queries)

        # Adapt top_k to question type — broad asks need more recall to be COMPLETE.
        eff_k = _effective_top_k(question, self.top_k)

        all_hits: dict[str, Hit] = {}
        for sub_q in sub_queries:
            # Expand "Q2 FY26" → "...quarter ended 30-09-2025"
            search_q = expand_fiscal_terms(sub_q)

            if rerank.is_enabled():
                candidates = self.retriever.search(search_q, k=max(RERANK_CANDIDATE_K, eff_k * 3))
                sub_hits = rerank.rerank(search_q, candidates, top_k=eff_k)
            else:
                sub_hits = self.retriever.search(search_q, k=eff_k)
            
            for h in sub_hits:
                if h.chunk.citation not in all_hits or h.score > all_hits[h.chunk.citation].score:
                    all_hits[h.chunk.citation] = h

        hits = sorted(all_hits.values(), key=lambda x: x.score, reverse=True)[:eff_k * len(sub_queries)]

        # Refusal logic: if the best chunk is weak on both signals, don't even ask the LLM.
        refused = False
        if not hits or (
            max((h.dense_score for h in hits), default=0.0) < MIN_DENSE_SCORE
            and max((h.sparse_score for h in hits), default=0.0) < MIN_SPARSE_SCORE
        ):
            ans_text = llm.NOT_FOUND
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            refused = True
        else:
            ctx = format_context(hits)
            r = llm.answer(question=rewritten, context_block=ctx, history=self.history)
            ans_text = r["text"]
            usage = r["usage"]
            if "not found in the document" in ans_text.lower():
                ans_text = llm.NOT_FOUND
                refused = True

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": ans_text})

        return Turn(question=question, rewritten=rewritten, answer=ans_text,
                    hits=hits, usage=usage, refused=refused)
