"""LLM — dispatches to NVIDIA NIM or Google Gemini based on LLM_BACKEND.

Public API:
  - decompose_and_rewrite(history, question) -> List[str]   standalone queries
  - answer(question, context_block, history) -> {"text", "usage"}
  - answer_document_intelligence(...)        -> {"text", "usage"}
  - active_backend() / active_model()

Why temperature 0 everywhere:
  Retrieval has already decided what the answer is allowed to contain. Sampling
  would only introduce drift between the cited chunks and the prose around them,
  which is precisely the failure this system exists to prevent.

What ADVANCED_SYSTEM is doing:
  The TABLE_QUERY block is the load-bearing part. A financial column header
  encodes two things at once — a period ("30-09-2025") and a qualifier
  ("Quarter Ended", "Unaudited") — so the model is told to match the row label
  AND the column header before reading a cell. Picking the right row from the
  wrong column is the single most common source of wrong numbers in financial
  RAG, and it produces answers that look completely plausible.

  The FORBIDDEN list exists because hedge-then-speculate ("the document doesn't
  say X, however...") is how a grounded model leaks ungrounded content. Refusal
  has to be a binary decision or it isn't a guarantee.

Usage accounting:
  Both backends return {"text", "usage"} so callers can emit token metrics. The
  SDKs disagree on field names (prompt_tokens vs prompt_token_count), hence the
  getattr() defaults rather than direct attribute access.
"""
from __future__ import annotations

import os
import re
import time
from typing import List

from . import observability as _obs

NOT_FOUND = "Not found in the document"
NOT_FOUND_DOCUMENTS = "Not found in the document"

_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s")


ADVANCED_SYSTEM = """You are a strict document-grounded answering system.

You answer questions using ONLY the provided CONTEXT. You have NO other knowledge.

---

## ABSOLUTE RULES (NEVER VIOLATE)

1. **ANSWER ONLY FROM CONTEXT**: Every claim in your answer MUST come from the provided context.
   If the specific information the user asked for is NOT in the context, respond EXACTLY:
   "Not found in the document"

2. **NEVER DO THIS** (these are hallucination patterns you MUST avoid):
   - "I don't have information about X, but the document mentions Y..." → FORBIDDEN
   - "While X is not explicitly stated, here's what the document says..." → FORBIDDEN
   - "The document doesn't directly address X, however..." → FORBIDDEN
   - Answering a DIFFERENT question than what was asked → FORBIDDEN
   - Providing tangentially related information when the exact answer is missing → FORBIDDEN

3. **BINARY DECISION**: Either you can answer the question from the context, or you cannot.
   There is NO middle ground. No hedging. No "related information" filler.

4. **EXACT NUMBERS**: Never round, approximate, or rephrase financial figures.
   Copy numbers exactly as they appear: "₹4,523 Cr" not "approximately ₹4,500 Cr".

5. **CITATIONS REQUIRED**: Every fact must have a citation [doc_id:pX:cY].
   If you cannot cite it, you cannot say it.

---

## RESPONSE STRATEGY

Determine the query intent internally (do NOT output this):

- **FACT_LOOKUP**: Extract exact values. Be precise.
- **SUMMARIZATION**: Structured summary preserving all key numbers.
- **COMPARISON**: Clearly separate entities. Use tables if helpful.
- **TABLE_SUMMARY**: When the user asks for a summary of a table or financial statement:
  * Produce a comprehensive structured summary of ALL data in the table.
  * Use the "Structured Table Relationships" (format: "<row_label> | <column_header>: <value>") to enumerate all key line items.
  * Group by period/column (e.g. "Quarter Ended 30-09-2025", "Half Year Ended 30-09-2025").
  * Include ALL major rows: Revenue, Expenses, Profit, EPS, etc.
  * Use a markdown table or organized bullet points for clarity.
  * Cite the table chunk: [doc_id:pX:cY].
- **TABLE_QUERY**: Interpret tabular data. CRITICAL RULES:
  * Every column header in a financial table encodes BOTH a period (e.g. "30-09-2025") AND a qualifier (e.g. "Quarter Ended", "Unaudited").
  * To answer "Finance cost for quarter ended 30-09-2025": find the ROW whose label is "Finance cost" (or "(d) Finance cost"), then read the cell in the COLUMN whose header contains "30-09-2025" AND "Quarter Ended". Do NOT use values from any other column.
  * Never substitute a value from a different row even if the column date matches.
  * Never substitute a value from a different column even if the row label matches.
  * The "Structured Table Relationships" in the context encode this as: "<row_label> | <column_header>: <value>". Use these facts verbatim.
- **ANALYSIS**: Explain trends/causes using ONLY evidence from context.

---

## OUTPUT FORMAT

## Answer

[Your grounded response with inline citations]

## Supporting Evidence

* [doc_id:pX:cY] — key quote or data point

---

## MULTI-DOCUMENT

If context spans multiple documents, use per-document citations:
[doc1:p12:c3], [doc2:p5:c1]

---

## LENGTH

- Short queries → concise (3–5 bullets)
- Broad queries → comprehensive with sections
- Always prefer precision over verbosity"""


REWRITE_SYSTEM = """Rewrite the user's latest input into one or more STANDALONE search
queries suitable for retrieval. Resolve pronouns and references using the chat history.
CRITICAL: If the user asks multiple distinct questions (e.g., "What is X? And what is Y?"), split them into separate standalone queries.
Return each standalone query on a new line. Do NOT output anything else. Do NOT include bullet points, numbering, or quotes.
If it is a single question, return just one line."""


def active_backend() -> str:
    """Generation provider from LLM_BACKEND ('nvidia' default)."""
    return os.getenv("LLM_BACKEND", "nvidia").strip().lower()


def active_model() -> str:
    """Chat model id for the active backend."""
    if active_backend() == "nvidia":
        # Default moved off llama-3.3-70b in Aug 2026: that model is heavily
        # congested on the NIM free tier (measured 32-46s or outright timeout,
        # reproduced across two API keys), while 3.1-70b answers the same
        # grounded-extraction prompt correctly in ~1-3s. Override per deployment
        # with NVIDIA_CHAT_MODEL.
        return os.getenv("NVIDIA_CHAT_MODEL", "meta/llama-3.1-70b-instruct")
    return os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")


def _maybe_wait_429(msg: str) -> bool:
    """Sleep and report True if `msg` is a rate-limit error; parses retryDelay."""
    if "429" not in msg and "RESOURCE_EXHAUSTED" not in msg and "rate" not in msg.lower():
        return False
    m = _RETRY_DELAY_RE.search(msg)
    wait = float(m.group(1)) if m else 30.0
    time.sleep(max(wait + 1.0, 5.0))
    return True


def _nvidia_client():
    """OpenAI-compatible client pointed at NVIDIA NIM. Raises if the key is unset."""
    from openai import OpenAI
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set. Get one at https://build.nvidia.com")
    base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def _nvidia_generate(system: str, user: str, *, temperature: float = 0.0) -> dict:
    """One completion via NIM, with retries. Returns {"text", "usage"}.

    Retries indiscriminately, including on non-retryable errors such as a bad
    key — wasteful but harmless, and it keeps transient 5xx from surfacing.
    """
    client = _nvidia_client()
    model = active_model()
    last: Exception | None = None
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=2048,
            )
            text = (resp.choices[0].message.content or "").strip()
            u = getattr(resp, "usage", None)
            usage = {
                "prompt_tokens": getattr(u, "prompt_tokens", 0) if u else 0,
                "completion_tokens": getattr(u, "completion_tokens", 0) if u else 0,
                "total_tokens": getattr(u, "total_tokens", 0) if u else 0,
            }
            return {"text": text, "usage": usage}
        except Exception as e:
            last = e
            if _maybe_wait_429(str(e)):
                continue
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"NVIDIA generate failed: {last}")


def _gemini_client():
    """Google GenAI client. Raises if GEMINI_API_KEY is unset."""
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Get one at https://aistudio.google.com/apikey")
    return genai.Client(api_key=api_key)


def _gemini_generate(system: str, user: str, *, temperature: float = 0.0) -> dict:
    """One completion via Gemini. Returns {"text", "usage"}.

    Gemini takes the system prompt as `system_instruction` rather than a
    message role, and names its usage fields differently — both normalised here
    so callers cannot tell the backends apart.
    """
    from google.genai import types
    client = _gemini_client()
    model = active_model()
    last: Exception | None = None
    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                ),
            )
            text = (resp.text or "").strip()
            um = getattr(resp, "usage_metadata", None)
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0) if um else 0,
                "completion_tokens": getattr(um, "candidates_token_count", 0) if um else 0,
                "total_tokens": getattr(um, "total_token_count", 0) if um else 0,
            }
            return {"text": text, "usage": usage}
        except Exception as e:
            last = e
            if _maybe_wait_429(str(e)):
                continue
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"Gemini generate failed: {last}")


def _generate(system: str, user: str, *, temperature: float = 0.0) -> dict:
    """Returns {'text': str, 'usage': {prompt_tokens, completion_tokens, total_tokens}}.

    Wall-clock latency is recorded per backend so a congested upstream model
    shows up in metrics instead of looking like a frozen UI.
    """
    backend = active_backend()
    started = time.perf_counter()
    try:
        if backend == "nvidia":
            return _nvidia_generate(system, user, temperature=temperature)
        if backend == "gemini":
            return _gemini_generate(system, user, temperature=temperature)
        raise ValueError(f"Unknown LLM_BACKEND: {backend}")
    finally:
        elapsed = time.perf_counter() - started
        _obs.llm_duration_seconds.labels(backend).observe(elapsed)
        if elapsed > 20.0:
            _obs.log.warning("slow LLM response",
                             extra={"llm_backend": backend, "model": active_model(),
                                    "seconds": round(elapsed, 1)})


def _generate_text(system: str, user: str, *, temperature: float = 0.0) -> str:
    """Convenience: just the text (used by rewriter)."""
    return _generate(system, user, temperature=temperature)["text"]


def decompose_and_rewrite(history: List[dict] | None, question: str) -> List[str]:
    """Turn a conversational question into standalone retrieval queries.

    Does two jobs at once: resolves references against history ("what about the
    next quarter" -> an explicit query), and splits genuinely multi-part
    questions so each half can be retrieved for separately. Retrieval has no
    conversational memory, so an unresolved pronoun retrieves nothing.

    Falls back to a crude regex split if the model call fails — a degraded
    query still beats no answer.
    """
    convo = ""
    if history:
        convo = "Chat history:\n" + "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[-6:]) + "\n\n"
    prompt = f"{convo}Latest question: {question}\n\nStandalone queries (one per line):"
    try:
        out = _generate_text(REWRITE_SYSTEM, prompt, temperature=0.0)
        lines = [line.strip(' "-*\'') for line in out.splitlines() if line.strip()]
        return lines if lines else [question]
    except Exception:
        return [q.strip() for q in re.split(r'\?|\n| and ', question) if q.strip()] or [question]


def answer(question: str, context_block: str, history: List[dict] | None = None) -> dict:
    """Generate a grounded answer.

    Returns:
      { "text": str, "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int} }
    """
    history_str = ""
    if history:
        recent = history[-4:]
        history_str = "Recent chat:\n" + \
            "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent) + "\n\n"

    user = (
        f"{history_str}"
        f"USER QUERY:\n{question}\n\n"
        f"---\n\n"
        f"CONTEXT:\n{context_block}"
    )
    return _generate(ADVANCED_SYSTEM, user, temperature=0.0)


def answer_document_intelligence(
    question: str,
    context_block: str,
    intent: str,
    history: List[dict] | None = None,
) -> dict:
    """Generate a multi-document, citation-grounded intelligence answer."""
    history_str = ""
    if history:
        recent = history[-4:]
        history_str = "Recent chat:\n" + \
            "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent) + "\n\n"

    user = (
        f"{history_str}"
        f"USER QUERY:\n{question}\n\n"
        f"---\n\n"
        f"CONTEXT:\n{context_block}"
    )
    return _generate(ADVANCED_SYSTEM, user, temperature=0.0)
