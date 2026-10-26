"""Query routing and answer generation for hierarchical multi-document RAG.

The decision layer. Retrieval primitives live in retriever.py and
hierarchical.py; this module decides which of them a given question needs, how
much context to spend on it, and whether the evidence justifies answering at all.

Routing, in the order the checks actually run:

  1. is_casual_query()   "hi", "thanks"        -> canned reply, no retrieval
  2. needs_summary()     "summarise page 19"   -> Qwen summary pipeline
  3. route_query()       everything else       -> factual / summary / comparison

  These are two independent routers, not one: needs_summary() picks a summary
  LEVEL (page / document / pre) while route_query() picks an INTENT. They were
  added at different times and their responsibilities overlap — collapsing them
  into a single classifier is the obvious next cleanup in this file.

Why page-number questions bypass embedding search:
  "Summarise page 19" is semantically almost contentless. Its embedding is no
  closer to page 19 than to page 20, because the page number is a reference, not
  a topic. _direct_page_lookup() pulls those chunks by page number instead of
  searching for them — retrieval cannot solve a lookup problem wearing the
  costume of a question.

Why merged hits are sorted by dense_score, not by the fused RRF score:
  RRF scores are only meaningful within a single ranking. When a question
  decomposes into several sub-queries each produces its own RRF scale, and those
  numbers are not comparable to each other. dense_score is an absolute cosine
  value with the same meaning everywhere, so the cross-sub-query merge sorts on
  that instead.

Context budget:
  MAX_CONTEXT_CHARS caps what reaches the model. format_hierarchical_context()
  spends the budget top-down — the strongest hits are quoted at length and the
  remainder share what is left — so a long tail of weak chunks can never crowd
  out the evidence the answer actually rests on.

Degradation path:
  Every external dependency has a fallback. Qwen unavailable -> answer from raw
  context. Primary LLM unavailable -> _extractive_fallback() returns the
  retrieved evidence with its citations intact. The system becomes less fluent
  as dependencies fail, never less grounded, and never hands the user a 500.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from . import qwen_summarizer

from . import llm
from .chat import expand_fiscal_terms
from .hierarchical import CHUNK_LEVEL, DOCUMENT_LEVEL, PAGE_LEVEL, MultiDocumentRetriever
from .retriever import Hit


FACTUAL = "factual"
SUMMARY = "summary"
COMPARISON = "comparison"
PRESUMMARY = "presummary"
CASUAL = "casual"

MIN_DENSE_SCORE = 0.30
MIN_SPARSE_SCORE = 0.60

# Maximum total context chars sent to the LLM.  Keeps prompts inside model
# context windows and reduces noise that causes hallucination.
MAX_CONTEXT_CHARS = 12000

# Higher budget for table-centric queries — a single financial statement table
# can be 20K+ chars with all its structured relationships.
MAX_TABLE_CONTEXT_CHARS = 24000

_COMPARISON_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|difference|different|similar|similarities|"
    r"across documents|between documents|which document|higher|lower|more than|less than)\b",
    re.IGNORECASE,
)
_SUMMARY_RE = re.compile(
    r"\b(summary|summarize|summarise|overview|highlights|key points|document summary|"
    r"page summary|trends|risks|segment analysis|financial metrics|business metrics)\b",
    re.IGNORECASE,
)
_BROAD_RE = re.compile(r"\b(all|every|list|outline|explain|summarize|summarise|overview|compare)\b", re.IGNORECASE)
_DOC_LEVEL_RE = re.compile(
    r"\b(document|file|pdf|report|deck|overall|entire|full|complete|executive|overview)\b",
    re.IGNORECASE,
)
_PAGE_NUM_RE = re.compile(r"page\s*(?:number\s*)?(?:no\.?\s*)?(\d+)", re.IGNORECASE)
_TABLE_SUMMARY_RE = re.compile(
    r"\b(?:summary|summarize|summarise|overview)\b.*\b(?:table|statement|balance\s*sheet|p\s*[&]\s*l|profit\s*and\s*loss)\b"
    r"|\b(?:table|statement|balance\s*sheet)\b.*\b(?:summary|summarize|summarise|overview)\b",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(r"\[[^\]\s]+:p\d+:c\d+\]")
_LOOSE_CITATION_RE = re.compile(r"\[[^\]\s]*:p(?P<page>\d+):c(?P<chunk>\d+)\]")

# Casual / greeting detection — short-circuit before retrieval
_CASUAL_RE = re.compile(
    r"^(?:hi|hello|hey|hii+|howdy|good\s*(?:morning|afternoon|evening|day)|"
    r"yo|sup|what'?s?\s*up|how\s*(?:are|r)\s*(?:you|u|ya)|"
    r"thanks?(?:\s*you)?|thank\s*u|bye|goodbye|see\s*ya|"
    r"ok(?:ay)?|cool|nice|great|awesome|good|fine|"
    r"who\s*(?:are|r)\s*(?:you|u)|what\s*(?:can|do)\s*(?:you|u)\s*do)[\s?!.]*$",
    re.IGNORECASE,
)

_CASUAL_RESPONSE = (
    "Hello! I'm your document intelligence assistant. "
    "Ask me any question about the uploaded documents — "
    "financial metrics, summaries, comparisons, or specific data points."
)

# Hallucination patterns — if the LLM hedges then speculates, hard refuse.
_HALLUCINATION_HEDGES = [
    r"(?:I\s+)?(?:don'?t|do\s+not)\s+have\s+(?:specific\s+)?information\s+(?:about|on|regarding)",
    r"(?:the\s+)?document\s+does(?:n'?t|\s+not)\s+(?:explicitly\s+)?(?:mention|state|contain|provide|address|include)",
    r"not\s+(?:explicitly|directly|specifically)\s+(?:mentioned|stated|provided|found|available|addressed)",
    r"(?:while|although|though)\s+(?:the\s+)?(?:exact|specific|direct)\s+.{0,40}\s+(?:is\s+)?not\s+(?:found|mentioned|stated|available)",
    r"(?:I\s+)?(?:couldn'?t|could\s+not|cannot|can'?t)\s+find\s+(?:specific|exact|direct)",
    r"(?:this|the)\s+(?:information|data|detail)\s+(?:is\s+)?not\s+(?:available|present|found)",
    r"no\s+(?:specific|direct|explicit)\s+(?:information|data|mention|reference)",
    r"there\s+is\s+no\s+(?:mention|information|data|reference)\s+(?:of|about|regarding)",
    r"not\s+(?:covered|discussed|addressed)\s+in\s+(?:the\s+)?(?:provided\s+)?(?:context|document|text)",
    r"(?:however|but|nonetheless|nevertheless|that\s+said)[,;]\s+(?:the\s+document|here|based)",
]
_HALLUCINATION_RE = re.compile("|".join(_HALLUCINATION_HEDGES), re.IGNORECASE)


def is_casual_query(question: str) -> bool:
    """Return True if the question is a greeting / chitchat, not a document query."""
    cleaned = question.strip().rstrip("?!. ")
    if len(cleaned) > 80:
        return False  # real questions are longer
    return bool(_CASUAL_RE.match(cleaned))


@dataclass
class IntelligenceTurn:
    """One exchange: question, routing decision, evidence, answer and token cost."""
    question: str
    rewritten: str
    answer: str
    intent: str
    hits: List[Hit] = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    refused: bool = False


def route_query(question: str) -> str:
    """Classify a question as comparison, summary or factual.

    Regex rather than a model call: routing runs on every question, so it has
    to be free and deterministic. Order matters — comparison is checked first
    because "compare the summary of X and Y" is a comparison, not a summary.
    """
    if _COMPARISON_RE.search(question):
        return COMPARISON
    if _SUMMARY_RE.search(question):
        return SUMMARY
    return FACTUAL


def format_hierarchical_context(hits: List[Hit], max_total_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Format retrieved hits into a context block, respecting a total char budget.

    Dynamically adjusts per-hit length so the total stays within the budget,
    preventing context overflow that dilutes answer quality.
    """
    if not hits:
        return ""
    # Allocate per-hit budget, with a floor so each hit is meaningful
    per_hit = max(400, max_total_chars // len(hits))

    parts: List[str] = []
    total = 0
    for hit in hits:
        chunk = hit.chunk
        snippet = chunk.text
        if len(snippet) > per_hit:
            snippet = snippet[:per_hit].rstrip() + " ..."
        metadata = [
            chunk.citation,
            f"Document: {chunk.doc_id or 'unknown'}",
            f"Page: {chunk.page}",
            f"Content Type: {chunk.content_type or chunk.kind}",
        ]
        if chunk.section:
            metadata.append(f"Section: {chunk.section}")
        block = "\n".join(metadata) + "\n\n" + snippet
        # Stop adding if we would exceed total budget
        if total + len(block) > max_total_chars and parts:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)


def _format_table_priority_context(
    hits: List[Hit], max_total_chars: int = MAX_TABLE_CONTEXT_CHARS,
) -> str:
    """Format context giving table chunks the lion's share of the budget.

    A single financial statement table chunk can be 20K+ chars because it
    includes all structured row/column/value relationships.  The default
    even-split budget truncates these to ~1.5K, destroying the data the LLM
    needs to summarize.  This formatter:
      1. Gives each table chunk up to 80% of the total budget.
      2. Fills remaining budget with non-table hits.
    """
    if not hits:
        return ""

    table_hits = [h for h in hits if h.chunk.kind == "table"]
    other_hits = [h for h in hits if h.chunk.kind != "table"]

    parts: List[str] = []
    total = 0
    table_budget = int(max_total_chars * 0.80)

    # --- Table chunks first (full or nearly-full text) ---
    for hit in table_hits:
        chunk = hit.chunk
        snippet = chunk.text
        if len(snippet) > table_budget:
            snippet = snippet[:table_budget].rstrip() + " ..."
        metadata = [
            chunk.citation,
            f"Document: {chunk.doc_id or 'unknown'}",
            f"Page: {chunk.page}",
            f"Content Type: {chunk.content_type or chunk.kind}",
        ]
        if chunk.section:
            metadata.append(f"Section: {chunk.section}")
        block = "\n".join(metadata) + "\n\n" + snippet
        if total + len(block) > max_total_chars and parts:
            break
        parts.append(block)
        total += len(block)

    # --- Remaining budget for non-table context ---
    remaining = max_total_chars - total
    if remaining > 400 and other_hits:
        per_other = max(400, remaining // len(other_hits))
        for hit in other_hits:
            chunk = hit.chunk
            snippet = chunk.text
            if len(snippet) > per_other:
                snippet = snippet[:per_other].rstrip() + " ..."
            metadata = [
                chunk.citation,
                f"Document: {chunk.doc_id or 'unknown'}",
                f"Page: {chunk.page}",
                f"Content Type: {chunk.content_type or chunk.kind}",
            ]
            if chunk.section:
                metadata.append(f"Section: {chunk.section}")
            block = "\n".join(metadata) + "\n\n" + snippet
            if total + len(block) > max_total_chars and parts:
                break
            parts.append(block)
            total += len(block)

    return "\n\n---\n\n".join(parts)


class DocumentIntelligenceRAG:
    """Multi-document RAG with hierarchical retrieval and intent routing.

    Holds conversation history and a per-session summary cache, so repeated
    questions about the same pages do not pay for summarisation twice.
    """
    def __init__(self, retriever: MultiDocumentRetriever, top_k: int = 10):
        """Bind a multi-document retriever and set default recall width."""
        self.retriever = retriever
        self.top_k = top_k
        self.history: List[dict] = []
        # Session-level cache for Qwen summaries — avoids re-calling Qwen
        # when multiple questions touch the same pages/documents.
        self._summary_cache: dict[str, str] = {}

    def reset(self) -> None:
        """Clear history and the summary cache."""
        self.history.clear()
        self._summary_cache.clear()

    def ask(self, question: str, doc_ids: Optional[Iterable[str]] = None) -> IntelligenceTurn:
        """Answer one question across the in-scope documents.

        The main entry point, and the place every routing decision converges.
        Order of the checks below is deliberate: cheapest exits first, so a
        greeting never touches retrieval and a page-number lookup never pays
        for an embedding search.

        Args:
            question: the user's raw question, unmodified.
            doc_ids:  restrict retrieval to these documents; None searches all.

        Returns an IntelligenceTurn carrying the answer, the evidence behind it,
        the routed intent and token usage — enough for the caller to render
        citations and emit metrics without re-deriving anything.
        """
        # ---- Casual query short-circuit ----
        # Greetings like "hi", "hello", "how are you" should get a friendly
        # response without hitting retrieval or the LLM at all.
        if is_casual_query(question):
            return self._record(
                question, question, _CASUAL_RESPONSE, CASUAL, [], {}, False,
            )

        scoped_retriever = self.retriever.filtered(doc_ids)
        if not scoped_retriever.doc_ids:
            return self._record(question, question, llm.NOT_FOUND_DOCUMENTS, FACTUAL, [], {}, True)

        sub_queries = llm.decompose_and_rewrite(self.history, question)
        rewritten = " | ".join(sub_queries)
        intent = route_query(rewritten)

        # --- Determine summary level from BOTH original and rewritten ---
        # The LLM rewriter often strips structural cues like "page number 19"
        # or "table STATEMENT OF...", so we check both and take the first match.
        summary_level = (
            qwen_summarizer.needs_summary(question)
            or qwen_summarizer.needs_summary(rewritten)
        )

        # --- Page-number summary: direct lookup (bypass embedding search) ---
        # Embedding search CANNOT reliably resolve "page 19" to page-19 content
        # because the query is semantically vague.  Instead we directly pull
        # all chunks from that page number + the pre-computed page summary.
        page_nums_from_q = _PAGE_NUM_RE.findall(question) or _PAGE_NUM_RE.findall(rewritten)
        if summary_level == "page" and page_nums_from_q:
            target_pages = {int(p) for p in page_nums_from_q}
            direct_hits = self._direct_page_lookup(target_pages, scoped_retriever)
            if direct_hits:
                answer, usage = self._handle_summary_query(
                    rewritten, intent, direct_hits, "page",
                )
                refused = answer.strip() == llm.NOT_FOUND_DOCUMENTS
                return self._record(question, rewritten, answer, intent, direct_hits, usage, refused)

        # --- Standard retrieval ---
        all_hits: dict[str, Hit] = {}
        for sub_q in sub_queries:
            search_q = expand_fiscal_terms(sub_q)
            sub_hits = self._retrieve(search_q, intent, scoped_retriever)
            for h in sub_hits:
                # Multi-doc RAG combines multiple hits, we dedup by citation
                if h.chunk.citation not in all_hits or h.dense_score > all_hits[h.chunk.citation].dense_score:
                    all_hits[h.chunk.citation] = h

        # Sort hits primarily by dense_score as RRF scores are isolated to their query
        hits = sorted(all_hits.values(), key=lambda x: x.dense_score, reverse=True)
        k_limit = self._effective_top_k(rewritten, intent) * len(sub_queries)
        hits = hits[:k_limit]

        refused = self._insufficient(hits)
        if refused:
            answer = llm.NOT_FOUND_DOCUMENTS
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        else:
            # --- Qwen summary pipeline (for non-page-number summary queries) ---
            if summary_level in ("page", "document"):
                answer, usage = self._handle_summary_query(
                    rewritten, intent, hits, summary_level,
                )
                refused = answer.strip() == llm.NOT_FOUND_DOCUMENTS
                return self._record(question, rewritten, answer, intent, hits, usage, refused)

            # --- Table summary: use table-priority context formatting ---
            is_table_summary = bool(
                _TABLE_SUMMARY_RE.search(question) or _TABLE_SUMMARY_RE.search(rewritten)
            )
            if is_table_summary:
                context = _format_table_priority_context(hits)
            else:
                context = format_hierarchical_context(hits)

            # Pre-summarize context via Qwen for analytical/complex queries.
            # NOTE: We return ONLY the condensed version (not summary + full
            # original) to keep context length under control and reduce noise.
            if summary_level == "pre":
                context = self._presummarize_context(rewritten, context)

            try:
                result = llm.answer_document_intelligence(
                    question=rewritten,
                    context_block=context,
                    intent=intent,
                    history=self.history,
                )
                answer = self._normalize_answer(result["text"], hits)
                usage = result["usage"]
                refused = answer.strip() == llm.NOT_FOUND_DOCUMENTS
            except Exception:
                answer = self._extractive_fallback(hits, intent)
                usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        return self._record(question, rewritten, answer, intent, hits, usage, refused)

    # ----- Qwen summary helpers -----

    def _handle_summary_query(
        self,
        question: str,
        intent: str,
        hits: List[Hit],
        level: str,
    ) -> tuple[str, dict]:
        """Route summary queries through Qwen3-4B for efficient summarization.

        Page-level: summarizes retrieved page-level chunks.
        Document-level: summarizes retrieved document-level + page-level chunks.
        Then passes the Qwen summary + original context to the primary LLM
        for final grounded answer with citations.

        Uses a session-level cache and pre-computed summaries from disk to
        avoid redundant Qwen calls.
        """
        context = format_hierarchical_context(hits)

        # --- Build cache key ---
        if level == "page":
            page_nums = re.findall(r'page\s*(?:number\s*)?(\d+)', question, re.I)
            page_key = ",".join(sorted(page_nums)) or ",".join(str(h.chunk.page) for h in hits[:3])
        else:
            page_key = "doc"
        doc_key = ",".join(sorted(set(h.chunk.doc_id for h in hits if h.chunk.doc_id))) or "unknown"
        cache_key = f"{level}:{doc_key}:{page_key}"

        # --- Check session cache first ---
        if cache_key in self._summary_cache:
            qwen_summary = self._summary_cache[cache_key]
        else:
            # --- Try loading from pre-computed summaries on disk ---
            qwen_summary = self._load_precomputed_summary(hits, level)

            # --- Fall back to Qwen API call ---
            if not qwen_summary:
                qwen_summary = self._generate_qwen_summary(question, hits, level)

            # --- Store in session cache ---
            if qwen_summary:
                self._summary_cache[cache_key] = qwen_summary

        if not qwen_summary:
            # No summary available — use standard retrieval path
            result = llm.answer_document_intelligence(
                question=question,
                context_block=context,
                intent=intent,
                history=self.history,
            )
            return self._normalize_answer(result["text"], hits), result["usage"]

        try:
            enriched_context = (
                f"=== SUMMARY (use for structure) ===\n"
                f"{qwen_summary}\n\n"
                f"=== ORIGINAL CONTEXT (use for citations) ===\n"
                f"{context}"
            )
            # Enforce context length budget
            if len(enriched_context) > MAX_CONTEXT_CHARS:
                enriched_context = enriched_context[:MAX_CONTEXT_CHARS]

            result = llm.answer_document_intelligence(
                question=question,
                context_block=enriched_context,
                intent=intent,
                history=self.history,
            )
            answer = self._normalize_answer(result["text"], hits)
            return answer, result["usage"]
        except Exception:
            # Fallback: standard path without summary
            result = llm.answer_document_intelligence(
                question=question,
                context_block=context,
                intent=intent,
                history=self.history,
            )
            return self._normalize_answer(result["text"], hits), result["usage"]

    def _load_precomputed_summary(self, hits: List[Hit], level: str) -> Optional[str]:
        """Try to load a pre-computed summary from storage/{doc_id}/summaries.json.

        These summaries are built deterministically during indexing, so they
        are free (no API call) and consistent.
        """
        if not hits:
            return None

        storage_root = Path(__file__).resolve().parent.parent / "storage"

        for hit in hits:
            doc_id = hit.chunk.doc_id
            if not doc_id:
                continue
            summary_path = storage_root / doc_id / "summaries.json"
            if not summary_path.exists():
                continue
            try:
                raw = json.loads(summary_path.read_text(encoding="utf-8"))

                if level == "document":
                    doc_sum = raw.get("document_summary", {})
                    # Reconstruct the markdown summary from the stored fields
                    parts = []
                    for key in ("key_highlights", "financial_business_metrics",
                                "segment_analysis", "trends", "risks"):
                        items = doc_sum.get(key, [])
                        if items:
                            header = key.replace("_", " ").title()
                            parts.append(f"### {header}")
                            parts.extend(f"* {item}" for item in items)
                    if parts:
                        return "\n".join(parts)

                elif level == "page":
                    target_pages = {hit.chunk.page for hit in hits}
                    page_sums = raw.get("page_summaries", [])
                    matching = [
                        ps for ps in page_sums
                        if ps.get("page") in target_pages
                    ]
                    if matching:
                        parts = []
                        for ps in matching:
                            parts.append(f"## Page {ps.get('page', '?')} — {ps.get('section', '')}")
                            for key in ("key_insights", "important_metrics", "data_interpretation"):
                                items = ps.get(key, [])
                                if items:
                                    parts.extend(f"* {item}" for item in items)
                        return "\n".join(parts)
            except Exception:
                continue
        return None

    def _generate_qwen_summary(self, question: str, hits: List[Hit], level: str) -> Optional[str]:
        """Generate a summary via Qwen API. Returns None if Qwen is unavailable."""
        try:
            if level == "page":
                page_nums = re.findall(r'page\s*(?:number\s*)?(\d+)', question, re.I)
                page_text = "\n\n".join(
                    f"--- Page {h.chunk.page} ---\n{h.chunk.text}"
                    for h in hits
                    if not page_nums or str(h.chunk.page) in page_nums
                )
                return qwen_summarizer.summarize_page(
                    page_text,
                    page=int(page_nums[0]) if page_nums else hits[0].chunk.page if hits else 0,
                    section=hits[0].chunk.section if hits else "",
                )
            else:  # document level
                doc_text = "\n\n".join(h.chunk.text for h in hits)
                doc_id = hits[0].chunk.doc_id if hits else ""
                return qwen_summarizer.summarize_document(doc_text, doc_id)
        except Exception:
            return None

    def _presummarize_context(self, question: str, context: str) -> str:
        """Use Qwen to condense context for complex analytical queries.

        Returns ONLY the condensed context (not summary + original) to keep
        total context length under control and reduce noise.
        Falls back to original context if Qwen is unavailable.
        """
        # Check session cache
        cache_key = f"pre:{hash(question)}:{hash(context[:200])}"
        if cache_key in self._summary_cache:
            return self._summary_cache[cache_key]

        try:
            compressed = qwen_summarizer.summarize_context(question, context)
            self._summary_cache[cache_key] = compressed
            return compressed
        except Exception:
            return context

    def _direct_page_lookup(
        self,
        target_pages: set[int],
        retriever: MultiDocumentRetriever,
    ) -> List[Hit]:
        """Directly retrieve all chunks belonging to specific page numbers.

        This bypasses embedding-based search entirely, which is critical for
        queries like "page number 19 summary" where the query has no semantic
        overlap with the page content.  We scan the chunk-level and page-level
        indexes for exact page-number matches.
        """
        hits: List[Hit] = []
        for hr in retriever.retrievers.values():
            # Chunk-level index: all raw chunks from those pages
            for chunk in hr.chunk_index.chunks:
                if chunk.page in target_pages:
                    hits.append(Hit(
                        chunk=chunk, score=1.0,
                        dense_score=1.0, sparse_score=1.0,
                    ))
            # Page-level index: pre-computed page summaries
            if hr.page_index:
                for chunk in hr.page_index.chunks:
                    if chunk.page in target_pages:
                        hits.append(Hit(
                            chunk=chunk, score=1.0,
                            dense_score=1.0, sparse_score=1.0,
                        ))
        return hits

    def _retrieve(self, query: str, intent: str, retriever: MultiDocumentRetriever) -> List[Hit]:
        """Retrieve for one sub-query, choosing tiers based on the routed intent."""
        k = self._effective_top_k(query, intent)
        if intent == COMPARISON:
            return retriever.search_levels(
                query,
                levels=[DOCUMENT_LEVEL, PAGE_LEVEL, CHUNK_LEVEL],
                k=k,
                per_level_k=max(4, k // 2),
            )
        if intent == SUMMARY:
            if _DOC_LEVEL_RE.search(query):
                levels = [DOCUMENT_LEVEL, PAGE_LEVEL, CHUNK_LEVEL]
            else:
                levels = [PAGE_LEVEL, CHUNK_LEVEL]
            return retriever.search_levels(query, levels=levels, k=k, per_level_k=max(4, k // 2))
        return retriever.search(query, level=CHUNK_LEVEL, k=k)

    def _effective_top_k(self, query: str, intent: str) -> int:
        """Widen recall for broad questions that need to enumerate rather than pinpoint."""
        base = max(1, self.top_k)
        if intent in (SUMMARY, COMPARISON) or _BROAD_RE.search(query):
            return max(base, 12)
        return base

    def _insufficient(self, hits: List[Hit]) -> bool:
        """Whether the evidence is too weak to answer at all.

        Requires BOTH signals to be below their floor. Either one alone clearing
        its threshold is enough to proceed, because a strong lexical match on an
        exact figure and a strong semantic match on a paraphrase are both valid
        grounds for answering.
        """
        if not hits:
            return True
        best_dense = max((hit.dense_score for hit in hits), default=0.0)
        best_sparse = max((hit.sparse_score for hit in hits), default=0.0)
        return best_dense < MIN_DENSE_SCORE and best_sparse < MIN_SPARSE_SCORE

    def _normalize_answer(self, answer: str, hits: List[Hit]) -> str:
        """Normalize LLM output. Detects hallucination-hedging patterns and
        enforces hard refusal instead of allowing "I don't know but..." answers.
        """
        text = (answer or "").strip()

        # Canonical refusal
        if "not found in the document" in text.lower():
            return llm.NOT_FOUND_DOCUMENTS
        if not text:
            return llm.NOT_FOUND_DOCUMENTS

        # --- Hallucination hedge detection ---
        # If the LLM says "I don't have info about X, but here's Y..." that's
        # a hallucination pattern.  Hard refuse instead of letting it through.
        if _HALLUCINATION_RE.search(text):
            return llm.NOT_FOUND_DOCUMENTS

        text = self._repair_citations(text, hits)

        # If no citations in the answer, we do NOT append evidence chunks.
        # That old behavior was itself a hallucination source — the LLM
        # would say something vague and then we'd tack on chunk snippets
        # making it look grounded when it wasn't.
        return text

    def _repair_citations(self, answer: str, hits: List[Hit]) -> str:
        """Rewrite citations so every one points at a chunk actually retrieved.

        Models routinely emit plausible-looking but wrong citation ids. Repairing
        them against the real hits is what makes the citations trustworthy rather
        than decorative — an unrepairable citation is dropped, not guessed.
        """
        by_page_chunk = {(hit.chunk.page, hit.chunk.chunk_id): hit.chunk.citation for hit in hits}

        def repl(match: re.Match) -> str:
            """Map one matched citation onto a real retrieved chunk, or drop it."""
            key = (int(match.group("page")), int(match.group("chunk")))
            return by_page_chunk.get(key, match.group(0))

        return _LOOSE_CITATION_RE.sub(repl, answer)

    def _extractive_fallback(self, hits: List[Hit], intent: str) -> str:
        """Answer from retrieved evidence alone when the LLM is unavailable.

        Degraded but still grounded and still cited — the user gets the relevant
        passages rather than a 500.
        """
        answer_lines = [
            "## Answer",
            "",
            "* Relevant evidence was retrieved, but the generation backend was unavailable. Use the cited evidence below for a grounded answer.",
            "",
            "## Evidence",
            "",
        ]
        answer_lines.extend(f"* {hit.chunk.citation} {self._short_note(hit)}" for hit in hits[:8])
        if intent == COMPARISON:
            answer_lines.extend(["", "## Comparison", "", "* Not found in the documents"])
        return "\n".join(answer_lines)

    def _short_note(self, hit: Hit) -> str:
        """Trim a chunk to a short quotable excerpt for the fallback answer."""
        text = re.sub(r"\s+", " ", hit.chunk.text).strip()
        return text[:260] + ("..." if len(text) > 260 else "")

    def _record(
        self,
        question: str,
        rewritten: str,
        answer: str,
        intent: str,
        hits: List[Hit],
        usage: dict,
        refused: bool,
    ) -> IntelligenceTurn:
        """Append the turn to history and return it."""
        usage = usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        return IntelligenceTurn(
            question=question,
            rewritten=rewritten,
            answer=answer,
            intent=intent,
            hits=hits,
            usage=usage,
            refused=refused,
        )
