"""Chunk enrichment for document-intelligence retrieval.

The goal is deterministic, cacheable understanding before embeddings:
tables become row/column/value facts, text chunks expose metric-bearing
sentences, and chart references are represented from surrounding text.

Everything here is regex and string work — no model calls. That is deliberate:
enrichment runs over every chunk of every document, so anything billed per token
would dominate indexing cost, and anything sampled would make indexing
non-reproducible. Same PDF in, same chunks out, every time.

The table relationships matter most. A markdown table row is nearly useless to
an embedding — "| Finance cost | 469.80 | 349.63 |" carries almost no semantic
signal. extract_table_relationships() flattens it into explicit facts of the form
"<row_label> | <column_header>: <value>", which restores the binding between a
number, the metric it measures and the period it belongs to. The system prompt in
llm.py refers to these by name and instructs the model to use them verbatim, so
the two files are co-designed and need to change together.

Known gap:
  Chunks are enriched with facts drawn from their own text, but never told where
  they sit in the document. A table fragment is embedded with no indication that
  it came from the H1 FY26 consolidated statement. Prepending a short section or
  page context string before embedding (the summaries in summarizer.py already
  compute one) is the highest-leverage retrieval improvement still available here.
"""
from __future__ import annotations

import re
from typing import Iterable, List

from .chunker import Chunk
from .ingest import Page


_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[$RsINR., ]{0,6})-?\d[\d,]*(?:\.\d+)?(?:\s?(?:%|x|bps|Cr|crore|mn|million|bn|billion|MW|GW|MT|MTPA|kWh|units?))?",
    re.IGNORECASE,
)
_GRAPH_RE = re.compile(r"\b(chart|graph|figure|trend|waterfall|growth|decline|increase|decrease|higher|lower)\b", re.IGNORECASE)
_PIPE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")


def _dedupe(items: Iterable[str], limit: int = 24) -> List[str]:
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
    """Split into sentences after collapsing whitespace."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _parse_markdown_table(table: str) -> tuple[list[str], list[list[str]]]:
    """Parse a pipe-delimited markdown table into (header, body).

    Handles tables produced by the multi-row header merger in chunker.py:
    - The separator line (| --- | ... |) is skipped.
    - A second non-numeric header row (qualifier row like 'Unaudited') is
      detected: if it contains no digits and all cells are text/empty it is
      merged into the header instead of treated as a data row.  This mirrors
      the same logic in _opendataloader_table_to_markdown so both paths agree.
    """
    rows: List[List[str]] = []
    for line in table.splitlines():
        match = _PIPE_ROW_RE.match(line)
        if not match:
            continue
        cells = [cell.strip() for cell in match.group(1).split("|")]
        # Drop separator rows (| --- | --- |)
        if cells and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells):
            continue
        if any(cells):
            rows.append(cells)
    if not rows:
        return [], []
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]

    # Multi-row header merge: if row[1] has no digits and looks like
    # qualifiers/annotations, merge it with row[0] to get full column labels.
    if len(rows) >= 2:
        second = rows[1]
        has_digit = any(re.search(r"\d", cell) for cell in second)
        all_qualifier = all(
            not cell or bool(re.fullmatch(r"[\s()A-Za-z,.₹Rs%/\-]+", cell.strip()))
            for cell in second
        )
        if not has_digit and all_qualifier:
            row0, row1 = rows[0], rows[1]
            merged = [
                " ".join(filter(None, [row0[i].strip(), row1[i].strip()])) or f"Column {i + 1}"
                for i in range(width)
            ]
            header = merged
            body = rows[2:]
        else:
            header, body = rows[0], rows[1:]
    else:
        header, body = rows[0], rows[1:]
    return header, body


def _has_number(text: str) -> bool:
    """Whether the text contains anything that reads as a figure."""
    return bool(_NUMBER_RE.search(text or ""))


def extract_text_metrics(text: str, limit: int = 16) -> List[str]:
    """Return metric sentences as value/context pairs without changing values."""
    metrics: List[str] = []
    for sentence in _split_sentences(text):
        values = _NUMBER_RE.findall(sentence)
        if not values:
            continue
        values_text = ", ".join(v.strip() for v in values if v.strip())
        metrics.append(f"{values_text}: {sentence}")
    return _dedupe(metrics, limit=limit)


def extract_graph_interpretations(text: str, limit: int = 8) -> List[str]:
    """Represent graph/chart meaning only from nearby extracted text."""
    return _dedupe(
        (sentence for sentence in _split_sentences(text) if _GRAPH_RE.search(sentence)),
        limit=limit,
    )


def extract_table_relationships(table: str, limit: int = 120) -> tuple[List[str], List[str]]:
    """Extract row/column/value facts from a markdown table.

    Generates unambiguous facts of the form:
        "<row_label> | <column_period>: <value>"

    The row_label is built from ALL leading columns that contain only text
    (no digits).  The column period comes from the merged header of each value
    column (after multi-row header merging in _parse_markdown_table).

    This fixes the previous bug where values from one row (e.g. 'Other Income:
    542.85') were attributed to a different row label (e.g. 'Finance cost')
    because the label-column boundary was hardcoded to idx <= 1, ignoring
    tables that have more leading label columns (Sr. No. + Particulars).
    """
    header, body = _parse_markdown_table(table)
    relationships: List[str] = []
    metrics: List[str] = []
    if not header:
        return metrics, relationships

    # Determine the label/value column boundary.
    # Strategy: header columns whose text contains no digits are label columns
    # (e.g. "Sr. No.", "Particulars"). The first header column that contains a
    # digit (a date like "30-09-2025") is the first value column.
    # This is more robust than scanning body rows, where Sr.No. cells ('1','2d')
    # would incorrectly be detected as numbers.
    first_value_col: int = len(header)
    for idx, h in enumerate(header):
        if _has_number(h):
            first_value_col = idx
            break

    # If no header column had a number (plain text headers), fall back to body
    # row scanning to find the first column that has numeric data.
    if first_value_col == len(header):
        for row in body:
            for idx, cell in enumerate(row):
                if cell and _has_number(cell):
                    first_value_col = min(first_value_col, idx)
                    break

    # Guard: if every column is a label (no value column found), default to
    # treating column 0 as the sole label and the rest as values.
    if first_value_col >= len(header):
        first_value_col = max(1, len(header) - 1)
    # Guard: at least one value column must exist.
    if first_value_col == 0:
        first_value_col = 1

    for row in body:
        # Build row label from all leading non-numeric columns.
        label_parts = [
            row[i].strip()
            for i in range(min(first_value_col, len(row)))
            if row[i].strip()
        ]
        row_label = " / ".join(label_parts) or (row[0].strip() if row else "row")

        # Emit one fact per value column, with bounds-safe header lookup.
        for idx in range(first_value_col, len(row)):
            cell = row[idx].strip() if idx < len(row) else ""
            if not cell or not _has_number(cell):
                continue
            col_label = (
                header[idx].strip()
                if idx < len(header) and header[idx].strip()
                else f"column {idx + 1}"
            )
            fact = f"{row_label} | {col_label}: {cell}"
            relationships.append(fact)
            metrics.append(f"{col_label}: {cell} ({row_label})")

    return _dedupe(metrics, limit=limit), _dedupe(relationships, limit=limit)


def detect_content_type(chunk: Chunk) -> str:
    """Classify a chunk so retrieval and prompting can treat it appropriately.

    More specific than Chunk.kind: distinguishes a financial table from a
    narrative paragraph that merely mentions numbers.
    """
    if chunk.kind == "table":
        return "table"
    if chunk.kind == "image":
        return "image"
    has_table = "|" in chunk.text and "\n|" in chunk.text
    has_numbers = _has_number(chunk.text)
    if has_table and has_numbers:
        return "mixed"
    return "text"


def _page_context_for_table(page_text: str) -> str:
    """Pull the surrounding page text that gives a table its meaning.

    A table alone rarely states its own period or basis ("Unaudited",
    "Consolidated") — that lives in the prose around it.
    """
    sentences = _split_sentences(page_text)
    selected = [
        s for s in sentences
        if _has_number(s) or re.search(r"\b(table|financial|revenue|income|profit|ebitda|metric|segment)\b", s, re.I)
    ]
    if not selected:
        selected = sentences[:3]
    return " ".join(selected[:5])[:1200].strip()


def enrich_chunks(chunks: List[Chunk], pages: List[Page]) -> List[Chunk]:
    """Mutate chunks with deterministic content understanding and enriched text."""
    page_texts = {page.page: page.text for page in pages}
    for chunk in chunks:
        content_type = detect_content_type(chunk)
        chunk.content_type = content_type

        if content_type == "table":
            metrics, relationships = extract_table_relationships(chunk.text)
            chunk.metrics = metrics
            chunk.relationships = relationships
            surrounding = _page_context_for_table(page_texts.get(chunk.page, ""))
            sections = [
                f"Content Type: table",
                f"Document: {chunk.doc_id or 'unknown'}",
                f"Page: {chunk.page}",
            ]
            if chunk.section:
                sections.append(f"Section: {chunk.section}")
            if surrounding:
                sections.append(f"Surrounding Context: {surrounding}")
            if relationships:
                sections.append("Structured Table Relationships:\n" + "\n".join(f"- {r}" for r in relationships))
            if metrics:
                sections.append("Key Values:\n" + "\n".join(f"- {m}" for m in metrics))
            sections.append("Original Table:\n" + chunk.text)
            chunk.text = "\n\n".join(sections)
            continue

        metrics = extract_text_metrics(chunk.text)
        graph_notes = extract_graph_interpretations(chunk.text)
        chunk.metrics = metrics
        chunk.relationships = graph_notes
        if metrics or graph_notes or chunk.section:
            prefix = [
                f"Content Type: {content_type}",
                f"Document: {chunk.doc_id or 'unknown'}",
                f"Page: {chunk.page}",
            ]
            if chunk.section:
                prefix.append(f"Section: {chunk.section}")
            if metrics:
                prefix.append("Important Metrics:\n" + "\n".join(f"- {m}" for m in metrics))
            if graph_notes:
                prefix.append("Graph or Trend Interpretation From Text:\n" + "\n".join(f"- {g}" for g in graph_notes))
            prefix.append("Source Text:\n" + chunk.text)
            chunk.text = "\n\n".join(prefix)

    return chunks
