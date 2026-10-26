"""Chunker: OpenDataLoader-backed semantic chunks with PyMuPDF fallback.

OpenDataLoader PDF is used as the primary parser for text/table chunking so
reading order, table structure, and semantic element boundaries survive into
retrieval. PyMuPDF page records are still accepted because ingestion uses them
for image extraction and as a fallback when OpenDataLoader is unavailable.
"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .ingest import Page


@dataclass
class Chunk:
    """One indexable unit of a document, plus everything needed to cite it.

    kind distinguishes how it was produced (text/table/image/summary);
    content_type is the finer classification enrichment.py assigns. Both matter
    at retrieval time — tables and summaries are formatted differently in the
    prompt.
    """
    chunk_id: int           # global, 0-indexed
    page: int               # 1-indexed source page
    text: str
    kind: str = "text"      # "text" | "table" | "image"
    image_file: Optional[str] = None  # filename relative to storage/{doc_id}/images/
    doc_id: str = ""
    section: str = ""
    content_type: str = ""  # "text" | "table" | "mixed" | "image" | summary levels
    metrics: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        """The [doc:pPAGE:cID] string the model is instructed to cite."""
        if self.doc_id:
            return f"[{self.doc_id}:p{self.page}:c{self.chunk_id}]"
        return f"[p{self.page}:c{self.chunk_id}]"


# Bump this when chunk construction changes so cached indexes rebuild.
CHUNKER_VERSION = "opendataloader-pdf-json-v2"

# Heuristic: ~4 chars/token. 900 chars ~= 220 tokens. Good for dense decks.
DEFAULT_CHUNK_CHARS = 900
DEFAULT_OVERLAP_CHARS = 150
MIN_CHUNK_CHARS = 80


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")
_ODL_TEXT_TYPES = {
    "paragraph",
    "heading",
    "caption",
    "list",
    "list item",
    "text block",
}
_ODL_SKIP_TYPES = {"header", "footer", "image"}


def normalize_pdf_text(text: str) -> str:
    """Normalize common PDF extraction artifacts before chunking/indexing."""
    if not text:
        return ""
    replacements = {
        "\u00c2\u00a0": " ",
        "\u00c2 ": " ",
        "\u00a0": " ",
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # A recurring OCR/font artifact in the sample PDFs turns ampersands into
    # "8-" or "8 -", e.g. "Property, Plant 8- Equipment".
    dash_chars = r"\-\u2010\u2011\u2013\u2014"
    text = re.sub(rf"(?<=\w)\s+8\s*[{dash_chars}]\s+(?=\w)", " & ", text)
    text = re.sub(rf"(?<=\w)\s+8\s*[{dash_chars}](?=[A-Z])", " & ", text)
    text = _WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def _split_paragraphs(text: str) -> List[str]:
    """Split text on blank lines into paragraph units."""
    text = normalize_pdf_text(text)
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _pack(paragraphs: List[str], max_chars: int, overlap: int) -> List[str]:
    """Greedily pack paragraphs into chunks <= max_chars."""
    chunks: List[str] = []
    buf = ""
    for paragraph in paragraphs:
        p = normalize_pdf_text(paragraph)
        if not p:
            continue
        if len(p) > max_chars:
            if buf:
                chunks.append(buf.strip())
                buf = ""
            sents = _SENT_SPLIT.split(p)
            sub = ""
            for sent in sents:
                if len(sub) + len(sent) + 1 <= max_chars:
                    sub = (sub + " " + sent).strip()
                else:
                    if sub:
                        chunks.append(sub.strip())
                    if len(sent) > max_chars:
                        for i in range(0, len(sent), max_chars):
                            chunks.append(sent[i:i + max_chars].strip())
                        sub = ""
                    else:
                        sub = sent
            if sub:
                chunks.append(sub.strip())
            continue

        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p) if buf else p
        else:
            chunks.append(buf.strip())
            buf = p
    if buf.strip():
        chunks.append(buf.strip())

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        tail = prev[-overlap:]
        sp = tail.find(" ")
        if 0 < sp < len(tail) - 1:
            tail = tail[sp:].strip()
        out.append((tail + "\n" + cur).strip() if tail else cur)
    return out


def chunk_pages(
    pages: List[Page],
    max_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_OVERLAP_CHARS,
    image_captions: Optional[dict] = None,
    image_filenames: Optional[dict] = None,
) -> List[Chunk]:
    """Build fallback chunks from PyMuPDF-extracted pages."""
    chunks: List[Chunk] = []
    cid = 0
    for page in pages:
        for table in page.tables:
            text = normalize_pdf_text(table)
            if not text:
                continue
            chunks.append(Chunk(
                chunk_id=cid,
                page=page.page,
                text=text,
                kind="table",
                doc_id=page.doc_id,
                section=page.section,
                content_type="table",
            ))
            cid += 1

        cid = _append_image_chunks(
            chunks,
            cid,
            page,
            image_captions=image_captions,
            image_filenames=image_filenames,
        )

        page_text = normalize_pdf_text(page.text)
        if not page_text or len(page_text) < MIN_CHUNK_CHARS:
            if page_text:
                chunks.append(Chunk(
                    chunk_id=cid,
                    page=page.page,
                    text=page_text,
                    kind="text",
                    doc_id=page.doc_id,
                    section=page.section,
                    content_type="text",
                ))
                cid += 1
            continue

        for piece in _pack(_split_paragraphs(page_text), max_chars=max_chars, overlap=overlap):
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            chunks.append(Chunk(
                chunk_id=cid,
                page=page.page,
                text=piece,
                kind="text",
                doc_id=page.doc_id,
                section=page.section,
                content_type="text",
            ))
            cid += 1
    return chunks


def chunk_pdf(
    pdf_path: str | Path,
    pages: List[Page],
    max_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_OVERLAP_CHARS,
    image_captions: Optional[dict] = None,
    image_filenames: Optional[dict] = None,
) -> List[Chunk]:
    """Build chunks from OpenDataLoader JSON, falling back to PyMuPDF chunks.

    The caller still passes `pages` because ingestion extracts images with
    PyMuPDF before chunking. This function mutates each page's `text` and
    `section` with OpenDataLoader's reading order when conversion succeeds, so
    summaries use the same parsed text as retrieval.
    """
    try:
        doc = _load_opendataloader_json(pdf_path)
        return _chunks_from_opendataloader(
            doc=doc,
            pages=pages,
            max_chars=max_chars,
            overlap=overlap,
            image_captions=image_captions,
            image_filenames=image_filenames,
        )
    except Exception as exc:
        print(f"OpenDataLoader chunking failed; falling back to PyMuPDF chunks: {exc}", flush=True)
        return chunk_pages(
            pages,
            max_chars=max_chars,
            overlap=overlap,
            image_captions=image_captions,
            image_filenames=image_filenames,
        )


def _load_opendataloader_json(pdf_path: str | Path) -> dict:
    """Run OpenDataLoader and return its structured JSON, or {} if unavailable."""
    import opendataloader_pdf

    pdf = Path(pdf_path)
    if not pdf.exists():
        raise FileNotFoundError(pdf)

    base_dir = Path.cwd() / ".opendataloader_tmp"
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        base_dir = pdf.parent

    tmp = base_dir / f"odl_pdf_{uuid.uuid4().hex}"
    tmp.mkdir(parents=True, exist_ok=False)
    try:
        out_dir = tmp
        opendataloader_pdf.convert(
            input_path=str(pdf),
            output_dir=str(out_dir),
            format="json",
            quiet=True,
            table_method="cluster",
            reading_order="xycut",
            image_output="off",
        )
        json_files = sorted(out_dir.glob("*.json"))
        if not json_files:
            raise RuntimeError("OpenDataLoader did not produce JSON output")
        return json.loads(json_files[0].read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            base_dir.rmdir()
        except Exception:
            pass


def _chunks_from_opendataloader(
    *,
    doc: dict,
    pages: List[Page],
    max_chars: int,
    overlap: int,
    image_captions: Optional[dict],
    image_filenames: Optional[dict],
) -> List[Chunk]:
    """Build chunks from OpenDataLoader's structured elements.

    Preferred over raw text extraction because element boundaries, reading
    order and table structure are already resolved by the parser.
    """
    page_by_no = {page.page: page for page in pages}
    elements_by_page: dict[int, List[dict]] = {page.page: [] for page in pages}
    for element in doc.get("kids", []):
        page_no = _element_page(element)
        if page_no <= 0:
            continue
        elements_by_page.setdefault(page_no, []).append(element)

    chunks: List[Chunk] = []
    cid = 0
    for page_no in sorted(elements_by_page):
        page = page_by_no.get(page_no)
        doc_id = page.doc_id if page else ""
        section = page.section if page else ""
        text_blocks: List[str] = []
        page_text_for_summary: List[str] = []
        table_fingerprints: set[str] = set()
        section_locked = False

        for element in elements_by_page[page_no]:
            kind = (element.get("type") or "").lower()
            if kind in _ODL_SKIP_TYPES:
                continue
            if kind == "heading":
                heading = normalize_pdf_text(_element_text(element))
                if heading and not section_locked and _is_section_heading(heading):
                    section = heading[:120]
                    section_locked = True
            if kind == "table":
                cid = _flush_text_chunks(
                    chunks, cid, page_no, doc_id, section, text_blocks, max_chars, overlap
                )
                text_blocks = []
                table_text = _opendataloader_table_to_markdown(element)
                if table_text:
                    table_fingerprints.add(_fingerprint(table_text))
                    for table_piece in _split_table_markdown(table_text, max_chars=max(max_chars * 3, 2600)):
                        chunks.append(Chunk(
                            chunk_id=cid,
                            page=page_no,
                            text=table_piece,
                            kind="table",
                            doc_id=doc_id,
                            section=section,
                            content_type="table",
                        ))
                        cid += 1
                    page_text_for_summary.append(table_text)
                continue

            if kind in _ODL_TEXT_TYPES:
                text = _format_opendataloader_text(element)
                if text:
                    text_blocks.append(text)
                    page_text_for_summary.append(text)

        cid = _flush_text_chunks(
            chunks, cid, page_no, doc_id, section, text_blocks, max_chars, overlap
        )

        if page:
            page.section = section
            if page_text_for_summary:
                page.text = "\n\n".join(page_text_for_summary)

            # OpenDataLoader occasionally emits complex financial statements as
            # reading-order text rather than table nodes. Preserve PyMuPDF's
            # table extraction as a rescue path without duplicating exact tables.
            for table in page.tables:
                table_text = normalize_pdf_text(table)
                if not table_text or _fingerprint(table_text) in table_fingerprints:
                    continue
                chunks.append(Chunk(
                    chunk_id=cid,
                    page=page.page,
                    text=table_text,
                    kind="table",
                    doc_id=page.doc_id,
                    section=page.section,
                    content_type="table",
                ))
                table_fingerprints.add(_fingerprint(table_text))
                cid += 1

            cid = _append_image_chunks(
                chunks,
                cid,
                page,
                image_captions=image_captions,
                image_filenames=image_filenames,
            )

    if not chunks:
        raise RuntimeError("OpenDataLoader produced no indexable chunks")
    return chunks


def _element_page(element: dict) -> int:
    """Page number for an element, defaulting to 1 when absent."""
    page = element.get("page number") or element.get("page")
    try:
        return int(page)
    except Exception:
        return 0


def _element_text(value) -> str:
    """Flatten an element's text, which may be a string or nested list."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_element_text(item) for item in value)
    if not isinstance(value, dict):
        return ""

    parts: List[str] = []
    content = value.get("content")
    if content:
        parts.append(str(content))
    for key in ("kids", "list items", "rows", "cells"):
        child = value.get(key)
        if child:
            parts.append(_element_text(child))
    return normalize_pdf_text(" ".join(part for part in parts if part))


def _format_opendataloader_text(element: dict) -> str:
    """Render one element as the text that will be embedded."""
    kind = (element.get("type") or "").lower()
    if kind == "list":
        items = []
        for i, item in enumerate(element.get("list items", []) or [], start=1):
            text = _element_text(item)
            if text:
                items.append(f"{i}. {text}")
        return normalize_pdf_text("\n".join(items))
    return _element_text(element)


def _is_section_heading(text: str) -> bool:
    """Heuristic: does this line look like a section heading worth tracking?"""
    normalized = normalize_pdf_text(text)
    if len(normalized) < 5:
        return False
    generic = {
        "sr. no.",
        "assets",
        "equity and liabilities",
        "current assets",
        "current liabilities",
        "financial assets",
        "financial liabilities",
    }
    return normalized.lower() not in generic


def _opendataloader_table_to_markdown(element: dict) -> str:
    """Convert a table element into a markdown pipe table."""
    rows = element.get("rows") or []
    width = int(element.get("number of columns") or 0)
    matrix: List[List[str]] = []
    for row in rows:
        cells = row.get("cells") or []
        if cells:
            width = max(width, max(int(cell.get("column number") or 1) for cell in cells))
    if width <= 0:
        return ""

    for row in rows:
        values = [""] * width
        for cell in row.get("cells") or []:
            col = max(1, int(cell.get("column number") or 1)) - 1
            text = _element_text(cell)
            if col < width:
                values[col] = text
        if any(values):
            matrix.append(values)

    if not matrix:
        return ""

    # Multi-row header detection (critical for financial statement tables).
    # Financial PDFs commonly use TWO header rows:
    #   Row 0: period group labels  e.g. "Quarter Ended" | "30-09-2025" | ...
    #   Row 1: qualifiers           e.g. "" | "(Unaudited)" | "(Audited)" | ...
    # When row 1 is NOT a data row (no numbers, all cells are qualifiers or
    # empty) we merge it with row 0 to build a single unambiguous header.
    # This ensures column labels like "Quarter Ended 30-09-2025 (Unaudited)"
    # are correctly bound to every value in that column.
    header_rows = 1
    if len(matrix) >= 2:
        second_row = matrix[1]
        # A row is a qualifier row if none of its cells contain digits.
        has_digit = any(re.search(r"\d", cell) for cell in second_row)
        # Also treat it as a sub-header if all non-empty cells look like
        # annotations: parenthetical qualifiers, "Rs. in Crores", "Crores", etc.
        all_qualifier = all(
            not cell or bool(re.fullmatch(r"[\s()A-Za-z,.₹Rs%/\-]+", cell.strip()))
            for cell in second_row
        )
        if not has_digit and all_qualifier:
            header_rows = 2

    if header_rows == 2:
        row0, row1 = matrix[0], matrix[1]
        merged_header = [
            " ".join(filter(None, [row0[i].strip(), row1[i].strip()])) or f"Column {i + 1}"
            for i in range(width)
        ]
        header = merged_header
        body = matrix[2:]
    else:
        header = matrix[0]
        body = matrix[1:]

    if not any(header):
        header = [f"Column {i + 1}" for i in range(width)]

    # Build a compact pipe-table where every column is unambiguously labeled.
    header_line = "| " + " | ".join(_escape_markdown_cell(cell) for cell in header) + " |"
    sep_line = "| " + " | ".join(["---"] * width) + " |"
    lines = [header_line, sep_line]
    for row in body:
        lines.append("| " + " | ".join(_escape_markdown_cell(cell) for cell in row) + " |")
    return normalize_pdf_text("\n".join(lines))


def _escape_markdown_cell(text: str) -> str:
    """Escape pipes and newlines so a cell cannot break the table grid."""
    return normalize_pdf_text(text).replace("|", "\\|")


def _split_table_markdown(text: str, max_chars: int) -> List[str]:
    """Split a large markdown table into smaller pieces.

    Each piece starts with the FULL header block (header row + separator),
    which may now be a merged multi-row header.  We detect the separator line
    (all `---` cells) and use everything up to and including it as the header
    block so split pieces always carry the correct column labels.
    """
    if len(text) <= max_chars:
        return [text]
    lines = text.splitlines()
    if len(lines) <= 4:
        return _pack([text], max_chars=max_chars, overlap=0)

    # Locate the separator line (| --- | --- | ...) to determine header block.
    sep_idx = 1  # default: standard single-row header
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped[1:-1].split("|")]
            if cells and all(re.fullmatch(r":?-{3,}:?", c) for c in cells if c):
                sep_idx = idx
                break

    header = lines[: sep_idx + 1]  # everything through and including "---" row
    pieces: List[str] = []
    buf = header[:]
    for line in lines[sep_idx + 1 :]:
        candidate = "\n".join(buf + [line])
        if len(candidate) > max_chars and len(buf) > len(header):
            pieces.append("\n".join(buf))
            buf = header + [line]
        else:
            buf.append(line)
    if len(buf) > len(header):
        pieces.append("\n".join(buf))
    return pieces or [text[:max_chars]]


def _flush_text_chunks(
    chunks: List[Chunk],
    cid: int,
    page: int,
    doc_id: str,
    section: str,
    blocks: List[str],
    max_chars: int,
    overlap: int,
) -> int:
    """Pack accumulated paragraphs into chunks and append them to the output."""
    paragraphs = [normalize_pdf_text(block) for block in blocks if normalize_pdf_text(block)]
    if not paragraphs:
        return cid
    for piece in _pack(paragraphs, max_chars=max_chars, overlap=overlap):
        if len(piece) < MIN_CHUNK_CHARS:
            continue
        chunks.append(Chunk(
            chunk_id=cid,
            page=page,
            text=piece,
            kind="text",
            doc_id=doc_id,
            section=section,
            content_type="text",
        ))
        cid += 1
    return cid


def _append_image_chunks(
    chunks: List[Chunk],
    cid: int,
    page: Page,
    *,
    image_captions: Optional[dict],
    image_filenames: Optional[dict],
) -> int:
    """Append captioned images as their own chunks, so charts are retrievable."""
    if not image_captions:
        return cid
    for img in page.images:
        cap = image_captions.get((page.page, img.index))
        if not cap:
            continue
        fname = (image_filenames or {}).get((page.page, img.index)) or img.filename
        # If the vision model extracted a markdown table from a chart/graph,
        # treat it as a table chunk so it flows through full table enrichment
        # (row/column/value facts, structured relationships, surrounding context).
        has_table = "|" in cap and "\n|" in cap
        chunk_kind = "table" if has_table else "image"
        chunks.append(Chunk(
            chunk_id=cid,
            page=page.page,
            text=normalize_pdf_text(f"[Image on page {page.page}] {cap}"),
            kind=chunk_kind,
            image_file=fname,
            doc_id=page.doc_id,
            section=page.section,
            content_type=chunk_kind,
        ))
        cid += 1
    return cid


def _fingerprint(text: str) -> str:
    """Content hash used to drop chunks duplicated across parsers."""
    return re.sub(r"[^a-z0-9]+", "", normalize_pdf_text(text).lower())[:500]
