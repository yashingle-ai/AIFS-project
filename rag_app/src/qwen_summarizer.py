"""Qwen-based summarization pipeline — second LLM path for summary tasks.

Uses HuggingFace Inference API with Qwen/Qwen3-4B-Instruct-2507 for efficient
summarization at page-level and document-level granularity.

Two modes:
  1. DIRECT SUMMARY — user explicitly asks for a summary; Qwen generates it.
  2. PRE-SUMMARY    — a non-summary query would benefit from condensed context;
                      Qwen summarises the retrieved context first, then the
                      primary LLM (NVIDIA/Gemini) uses the summary to answer.

Public API:
  - is_enabled()              -> bool
  - needs_summary(question)   -> "page" | "document" | None
  - summarize_page(text, page, section)   -> str
  - summarize_document(text, doc_id)      -> str
  - summarize_context(question, context)  -> str  (pre-summary for primary LLM)
"""
from __future__ import annotations

import os
import re
import time
from typing import Optional


# Configuration

_HF_BASE_URL = "https://router.huggingface.co/v1"
_HF_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def is_enabled() -> bool:
    """Qwen summariser is enabled only if HF_API_KEY is set."""
    return bool(os.getenv("HF_API_KEY", "").strip())


# Intent detection — does this query need summarization?

_PAGE_SUMMARY_RE = re.compile(
    r"\b(summarize\s+page|summary\s+of\s+page|page\s+(?:number\s+)?\d+\s+(?:summary|overview)|"
    r"what\s+is\s+on\s+page|key\s+points?\s+(?:from|on|of)\s+page|"
    r"explain\s+page|describe\s+page|summary\s+(?:for|of)\s+page\s+(?:number\s+)?\d+)\b",
    re.IGNORECASE,
)
_DOC_SUMMARY_RE = re.compile(
    r"\b(summarize|summarise|summary|overview|highlights|key\s+points?|"
    r"executive\s+summary|main\s+findings|overall|gist|brief|"
    r"document\s+summary|what\s+does\s+(?:this|the)\s+(?:document|report|file)\s+(?:say|contain|cover)|"
    r"give\s+me\s+(?:a\s+)?summary|tldr|tl;dr)\b",
    re.IGNORECASE,
)

# Topic-specific subjects that signal the user wants retrieval, NOT a generic
# full-page or full-document summary.  "Summarize the revenue trends" should
# use standard retrieval + primary LLM, not Qwen's generic summarizer.
_TOPIC_SPECIFIC_RE = re.compile(
    r"\b(revenue|income|profit|loss|ebitda|margin|cash|debt|segment|growth|capex|"
    r"volume|capacity|cost|risk|approval|investment|guidance|trend|market|"
    r"customer|project|order|pipeline|airport|road|energy|utility|"
    r"dividend|equity|asset|liability|tax|depreciation|amortization|"
    r"operating|capital|interest|expense|salary|employee|workforce|"
    r"passenger|cargo|traffic|toll|tariff|regulation|"
    r"acquisition|merger|subsidiary|joint\s+venture|partnership|"
    r"table|chart|graph|statement|balance\s+sheet|p&l|pnl|profit\s+and\s+loss)\b",
    re.IGNORECASE,
)

_NEEDS_PRESUMMARY_RE = re.compile(
    r"\b(explain\s+the\s+(?:trend|growth|decline|change|impact|significance|importance)|"
    r"what\s+(?:are|is)\s+the\s+(?:main|key|major|important)\s+(?:drivers?|factors?|reasons?|causes?)|"
    r"how\s+has\s+.+\s+(?:changed|evolved|performed)|"
    r"analyze|analyse|assessment|insight|implication|interpret|"
    r"elaborate|in\s+detail|comprehensive|deep\s+dive)\b",
    re.IGNORECASE,
)


def needs_summary(question: str) -> Optional[str]:
    """Determine if a query should be routed through the Qwen summary pipeline.

    Returns:
      - "page"     → page-level summary requested (explicit page reference)
      - "document" → document-level summary requested (generic, no topic)
      - "pre"      → query benefits from pre-summarized context
      - None       → standard retrieval path, no summary needed

    Topic-specific queries like "summarize the revenue trends" or "key points
    about EBITDA" return None so they go through standard retrieval + primary
    LLM instead of the generic Qwen summarizer.
    """
    if not is_enabled():
        return None

    # Page-specific summaries are always fine — they reference a page number
    if _PAGE_SUMMARY_RE.search(question):
        return "page"

    # Check if this is a generic document summary request
    if _DOC_SUMMARY_RE.search(question):
        # If the query also mentions a specific topic, it's NOT a generic
        # summary — route through standard retrieval instead.
        if _TOPIC_SPECIFIC_RE.search(question):
            return None  # topic-specific → standard retrieval path
        return "document"

    if _NEEDS_PRESUMMARY_RE.search(question):
        return "pre"
    return None


# HuggingFace / Qwen client

def _hf_client():
    """HuggingFace inference client. Raises if the token is unset."""
    from openai import OpenAI
    api_key = os.getenv("HF_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("HF_API_KEY not set. Get one at https://huggingface.co/settings/tokens")
    return OpenAI(api_key=api_key, base_url=_HF_BASE_URL)


def _qwen_generate(system: str, user: str, *, temperature: float = 0.3,
                    max_tokens: int = 1500) -> str:
    """Call Qwen3-4B via HuggingFace Inference API."""
    client = _hf_client()
    last: Exception | None = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=_HF_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last = e
            msg = str(e)
            if "429" in msg or "rate" in msg.lower():
                time.sleep(min(2 ** attempt * 5, 30))
            else:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Qwen/HuggingFace generate failed: {last}")


# Summary prompts

_PAGE_SUMMARY_SYSTEM = """You are a financial document summarizer. Generate a concise, structured summary of the given page content.

Rules:
1. Preserve ALL exact numbers, metrics, and financial values — never round or approximate
2. Organize into clear sections: Key Findings, Financial Metrics, Notable Items
3. Keep it concise but complete — no important data should be lost
4. If tables are present, extract the most important rows/values
5. Mention the section/topic if identifiable
6. Use bullet points for clarity"""

_DOC_SUMMARY_SYSTEM = """You are a financial document summarizer. Generate a comprehensive executive summary of the entire document.

Rules:
1. Cover: Business Overview, Financial Performance, Key Metrics, Segment Analysis, Risks & Outlook
2. Preserve ALL exact numbers — revenue, EBITDA, margins, growth rates
3. Highlight YoY/QoQ changes where mentioned
4. Note any forward-looking statements or guidance
5. Structure with clear markdown headers and bullet points
6. Be thorough but avoid redundancy — aim for a 300-500 word summary"""

_PRESUMMARY_SYSTEM = """You are a context compression assistant. The user has a specific question and retrieved context from financial documents.

Your job: Condense the context into a focused summary that preserves ONLY the information relevant to answering the question.

Rules:
1. Keep ALL exact numbers, dates, and financial values from relevant sections
2. Remove boilerplate, headers, and irrelevant metadata
3. Preserve source citations [doc:pX:cY] so the primary LLM can cite them
4. If tables are relevant, keep the key rows/columns
5. Output should be 40-60% shorter than the input while retaining all answer-relevant facts"""


def summarize_page(text: str, page: int, section: str = "") -> str:
    """Generate a structured summary of a single page's content."""
    if not text.strip():
        return f"Page {page}: No content to summarize."
    header = f"Page {page}"
    if section:
        header += f" — Section: {section}"
    prompt = f"{header}\n\n{text[:6000]}"
    try:
        return _qwen_generate(_PAGE_SUMMARY_SYSTEM, prompt)
    except Exception as e:
        return f"Summary generation failed for page {page}: {e}"


def summarize_document(text: str, doc_id: str = "") -> str:
    """Generate a comprehensive executive summary of the full document."""
    if not text.strip():
        return "No content to summarize."
    header = f"Document: {doc_id}" if doc_id else "Document"
    # For large documents, we take the first ~8000 chars and last ~2000 chars
    # to capture both the overview and conclusions
    if len(text) > 10000:
        body = text[:8000] + "\n\n[...middle content omitted for brevity...]\n\n" + text[-2000:]
    else:
        body = text
    prompt = f"{header}\n\n{body}"
    try:
        return _qwen_generate(_DOC_SUMMARY_SYSTEM, prompt, max_tokens=2000)
    except Exception as e:
        return f"Document summary generation failed: {e}"


def summarize_context(question: str, context: str) -> str:
    """Pre-summarize retrieved context to help the primary LLM answer better.

    Used when the question involves analysis, interpretation, or complex reasoning
    that benefits from condensed, focused context.
    """
    if not context.strip():
        return context
    prompt = f"QUESTION: {question}\n\n---\n\nRETRIEVED CONTEXT:\n{context[:12000]}"
    try:
        return _qwen_generate(_PRESUMMARY_SYSTEM, prompt, max_tokens=2000)
    except Exception as e:
        # If Qwen fails, return original context — never break the pipeline
        return context
