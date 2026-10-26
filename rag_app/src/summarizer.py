"""Deterministic hierarchical summaries for pages and documents.

These summaries are precomputed at ingestion time and embedded separately from
raw chunks. They intentionally preserve numeric strings as they appear in the
source text and remain extractive enough to avoid introducing unsupported facts.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List

from .chunker import Chunk
from .enrichment import extract_graph_interpretations, extract_text_metrics
from .ingest import Page


_BUSINESS_RE = re.compile(
    r"\b(revenue|income|profit|loss|ebitda|margin|cash|debt|segment|growth|capex|"
    r"volume|capacity|traffic|passenger|cargo|risk|approval|investment|cost|"
    r"guidance|trend|market|customer|project|order|pipeline)\b",
    re.IGNORECASE,
)
_RISK_RE = re.compile(r"\b(risk|uncertain|decline|loss|debt|delay|litigation|impairment|pressure|challenge)\b", re.IGNORECASE)
_SEGMENT_RE = re.compile(r"\b(segment|business|division|vertical|airport|road|energy|utility|subsidiary)\b", re.IGNORECASE)


@dataclass
class PageSummary:
    """Deterministic summary of a single page.

    Three separate fields rather than free prose, because each is retrieved
    differently: insights carry the narrative, metrics carry the numbers a
    lexical query will match on, and interpretations describe chart movement.
    """
    doc_id: str
    page: int
    section: str = ""
    key_insights: List[str] = field(default_factory=list)
    important_metrics: List[str] = field(default_factory=list)
    data_interpretation: List[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        """Citation string identifying this summary as page-level evidence."""
        return f"[{self.doc_id}:p{self.page}:summary]" if self.doc_id else f"[p{self.page}:summary]"

    def to_markdown(self) -> str:
        """Render as the markdown body that gets embedded into the page index."""
        lines = [f"## Page Summary [doc:{self.doc_id or 'unknown'} p:{self.page}]"]
        if self.section:
            lines.extend(["", f"Section: {self.section}"])
        lines.extend(["", "### Key Insights"])
        lines.extend(_bullets(self.key_insights))
        lines.extend(["", "### Important Metrics"])
        lines.extend(_bullets(self.important_metrics))
        lines.extend(["", "### Data Interpretation"])
        lines.extend(_bullets(self.data_interpretation))
        return "\n".join(lines).strip()


@dataclass
class DocumentSummary:
    """Report-level summary assembled from the page summaries.

    Highlights, segments, trends and risks are split apart so a question about
    risk does not have to out-rank every other sentence in the document.
    """
    doc_id: str
    pages: int
    key_highlights: List[str] = field(default_factory=list)
    financial_business_metrics: List[str] = field(default_factory=list)
    segment_analysis: List[str] = field(default_factory=list)
    trends: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        """Citation string identifying this summary as document-level evidence."""
        return f"[{self.doc_id}:p0:summary]" if self.doc_id else "[document:summary]"

    def to_markdown(self) -> str:
        """Render as the markdown body that gets embedded into the document index."""
        lines = ["## Document Summary"]
        if self.doc_id:
            lines.extend(["", f"Document: {self.doc_id}"])
        lines.extend(["", "### Key Highlights"])
        lines.extend(_bullets(self.key_highlights))
        lines.extend(["", "### Financial / Business Metrics"])
        lines.extend(_bullets(self.financial_business_metrics))
        lines.extend(["", "### Segment Analysis"])
        lines.extend(_bullets(self.segment_analysis))
        lines.extend(["", "### Trends"])
        lines.extend(_bullets(self.trends))
        lines.extend(["", "### Risks"])
        lines.extend(_bullets(self.risks))
        return "\n".join(lines).strip()


def _bullets(items: Iterable[str]) -> List[str]:
    """Markdown bullets, with an explicit placeholder when the list is empty."""
    values = [item for item in items if item]
    return [f"* {item}" for item in values] if values else ["* No extracted items"]


def _dedupe(items: Iterable[str], limit: int) -> List[str]:
    """Whitespace-normalise, drop duplicates, keep order, stop at `limit`."""
    out: List[str] = []
    seen: set[str] = set()
    for item in items:
        clean = " ".join(str(item).split())
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
        if len(out) >= limit:
            break
    return out


def _split_sentences(text: str) -> List[str]:
    """Split on sentence-ending punctuation after collapsing whitespace."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _ranked_insights(text: str, limit: int = 5) -> List[str]:
    """Pick the most informative sentences from a page.

    Scores each sentence by what makes it useful as retrievable evidence:
    business vocabulary, presence of digits, and movement words like "growth"
    or "decline". Very long sentences are penalised because they tend to be
    boilerplate, and anything scoring zero is dropped entirely.

    Extractive by construction — sentences are copied, never rewritten — so a
    summary can never introduce a fact the page does not contain.
    """
    scored: List[tuple[int, str]] = []
    for sentence in _split_sentences(text):
        if len(sentence) < 35:
            continue
        score = 0
        if _BUSINESS_RE.search(sentence):
            score += 3
        if any(ch.isdigit() for ch in sentence):
            score += 2
        if re.search(r"\b(increase|decrease|growth|decline|higher|lower|improved|reduced)\b", sentence, re.I):
            score += 2
        if len(sentence) > 220:
            score -= 1
        scored.append((score, sentence[:320]))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return _dedupe((sentence for score, sentence in scored if score > 0), limit=limit)


def _page_chunks(chunks: List[Chunk], page: int) -> List[Chunk]:
    """Text and table chunks for one page. Image chunks are excluded."""
    return [chunk for chunk in chunks if chunk.page == page and chunk.kind != "image"]


def build_page_summaries(pages: List[Page], chunks: List[Chunk]) -> List[PageSummary]:
    """Build one summary per page from that page's chunks and raw text.

    Runs at index time with no model call, so it is free, reproducible and
    cacheable. intelligence.py later reuses these instead of paying for a Qwen
    summarisation call.
    """
    summaries: List[PageSummary] = []
    for page in pages:
        page_chunks = _page_chunks(chunks, page.page)
        chunk_text = "\n".join(chunk.text for chunk in page_chunks)
        metrics = _dedupe(
            [metric for chunk in page_chunks for metric in chunk.metrics] + extract_text_metrics(page.text),
            limit=12,
        )
        interpretations = _dedupe(
            [rel for chunk in page_chunks for rel in chunk.relationships] + extract_graph_interpretations(page.text),
            limit=8,
        )
        insights = _ranked_insights(page.text + "\n" + chunk_text, limit=6)
        if not insights and page.text:
            insights = _dedupe(_split_sentences(page.text), limit=3)
        summaries.append(PageSummary(
            doc_id=page.doc_id,
            page=page.page,
            section=page.section,
            key_insights=insights,
            important_metrics=metrics,
            data_interpretation=interpretations,
        ))
    return summaries


def build_document_summary(doc_id: str, pages: List[Page], page_summaries: List[PageSummary]) -> DocumentSummary:
    """Roll page summaries up into one report-level summary.

    Segments, trends and risks are separated by keyword so each becomes its own
    retrievable facet rather than being buried in a single block of prose.
    """
    all_insights = [item for summary in page_summaries for item in summary.key_insights]
    all_metrics = [item for summary in page_summaries for item in summary.important_metrics]
    all_interpretations = [item for summary in page_summaries for item in summary.data_interpretation]
    segment_items = [item for item in all_insights + all_interpretations if _SEGMENT_RE.search(item)]
    trends = [item for item in all_insights + all_interpretations if re.search(r"\b(trend|growth|increase|decrease|decline|higher|lower|improved|reduced)\b", item, re.I)]
    risks = [item for item in all_insights + all_interpretations if _RISK_RE.search(item)]
    return DocumentSummary(
        doc_id=doc_id,
        pages=len(pages),
        key_highlights=_dedupe(all_insights, limit=14),
        financial_business_metrics=_dedupe(all_metrics, limit=20),
        segment_analysis=_dedupe(segment_items, limit=12),
        trends=_dedupe(trends, limit=12),
        risks=_dedupe(risks, limit=10),
    )


def summaries_to_chunks(page_summaries: List[PageSummary], doc_summary: DocumentSummary) -> tuple[List[Chunk], List[Chunk]]:
    """Wrap the summaries as Chunks so they can be embedded like any other text.

    Uses a high chunk_id offset (100000 / 200000) to keep summary ids from
    colliding with real chunk ids in the same citation namespace.
    """
    page_chunks = [
        Chunk(
            chunk_id=100000 + i,
            page=summary.page,
            text=summary.to_markdown(),
            kind="page_summary",
            doc_id=summary.doc_id,
            section=summary.section,
            content_type="page_summary",
            metrics=summary.important_metrics,
            relationships=summary.data_interpretation,
        )
        for i, summary in enumerate(page_summaries)
    ]
    doc_chunks = [
        Chunk(
            chunk_id=200000,
            page=0,
            text=doc_summary.to_markdown(),
            kind="document_summary",
            doc_id=doc_summary.doc_id,
            content_type="document_summary",
            metrics=doc_summary.financial_business_metrics,
            relationships=doc_summary.trends + doc_summary.risks,
        )
    ]
    return page_chunks, doc_chunks


def save_summaries(path: str | Path, page_summaries: List[PageSummary], doc_summary: DocumentSummary) -> None:
    """Persist both summary levels to summaries.json for later reuse."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "page_summaries": [asdict(summary) for summary in page_summaries],
        "document_summary": asdict(doc_summary),
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_summaries(path: str | Path) -> tuple[List[PageSummary], DocumentSummary]:
    """Read back what save_summaries() wrote."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    pages = [PageSummary(**item) for item in raw.get("page_summaries", [])]
    doc = DocumentSummary(**raw["document_summary"])
    return pages, doc
