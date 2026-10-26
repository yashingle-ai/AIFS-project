"""PDF ingestion: extracts page-aware text + tables using PyMuPDF.

Returns page-aware records with text, tables, layout metadata, and optional
document identifiers. We keep tables separately so the chunker can preserve them
as atomic units.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from .table_reconstruct import is_financial_results_page, reconstruct_financial_table


@dataclass
#This is a data structure describing an image extracted from a PDF
class ImageRef:
    """A meaningful image extracted from a PDF page."""
    page: int                # 1-indexed
    index: int               # per-page index (0, 1, 2, ...)
    bytes: bytes             # raw image bytes
    ext: str                 # 'png', 'jpeg', 'jpg', 'webp'
    width: int
    height: int

    @property
    def mime(self) -> str:
        """MIME type implied by the image extension."""
        e = self.ext.lower()
        if e in ("jpg", "jpeg"): return "image/jpeg"
        if e == "webp": return "image/webp"
        return "image/png"

    @property
    def filename(self) -> str:
        """Stable filename derived from page and index, used for storage and citations."""
        return f"p{self.page:03d}_i{self.index:02d}.{self.ext}"


@dataclass
class Page:
    """One extracted page: text, tables, images and layout metadata.

    Tables are kept separate from text rather than inlined, so the chunker can
    treat them as atomic units instead of letting a splitter cut through rows.
    """
    page: int  # 1-indexed
    text: str
    tables: List[str] = field(default_factory=list)
    images: List[ImageRef] = field(default_factory=list)
    doc_id: str = ""
    section: str = ""
    layout: List[dict] = field(default_factory=list)


_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")
_SECTION_RE = re.compile(r"^([A-Z][A-Za-z0-9&.,:/() \-]{4,90})$")


def _clean(text: str) -> str:
    """Normalise whitespace and strip artefacts from extracted text."""
    text = text.replace("\u00a0", " ")
    text = _WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def _extract_layout(page: fitz.Page) -> List[dict]:
    """Extract lightweight layout blocks from PyMuPDF's structured text dict."""
    blocks: List[dict] = []
    try:
        raw = page.get_text("dict", sort=True) or {}
    except Exception:
        return blocks

    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        lines: List[str] = []
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            txt = "".join(span.get("text", "") for span in spans).strip()
            if txt:
                lines.append(txt)
        text = _clean("\n".join(lines))
        if not text:
            continue
        blocks.append({
            "bbox": [round(float(v), 2) for v in block.get("bbox", [])[:4]],
            "text": text[:1000],
        })
    return blocks

# Which section does this page probably belong to?
def _guess_section(text: str, previous: str = "") -> str:
    """Best-effort section detection from prominent early lines."""
    lines = [ln.strip(" -:|") for ln in text.splitlines()[:12] if ln.strip()]
    for line in lines:
        if len(line) < 5 or len(line) > 90:
            continue
        if _SECTION_RE.match(line) and sum(ch.isalpha() for ch in line) >= 4:
            return line
        if line.isupper() and any(ch.isalpha() for ch in line):
            return line.title()
    return previous


def _merge_multirow_header(rows: List[List[str]], max_header_rows: int = 3) -> tuple[List[str], List[List[str]]]:
    """Detect and merge multi-row table headers into single compound column labels.

    Financial statement tables commonly have 2-3 header rows:
      Row 0: "Quarter Ended" | "Half Year Ended" | "Year Ended"  (group labels)
      Row 1: "30-09-2025"    | "30-06-2025"      | "31-03-2025"  (period dates)
      Row 2: "(Unaudited)"   | "(Unaudited)"     | "(Audited)"   (qualifiers)

    A row is a header row when it has NO financial values (4+-digit numbers or
    multi-decimal figures like '5,066.43' or '469.80').
    Date strings ('30-09-2025') and qualifiers ('(Unaudited)') count as header text.
    """
    _FIN_NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d{2,}")

    def _is_header_row(row: List[str]) -> bool:
        """Whether a row looks like part of a header rather than data."""
        return not any(_FIN_NUM.search(cell) for cell in row if cell)

    if not rows:
        return [], []

    header_row_count = 1
    for i in range(1, min(max_header_rows, len(rows))):
        if _is_header_row(rows[i]):
            header_row_count = i + 1
        else:
            break

    if header_row_count == 1:
        return rows[0], rows[1:]

    width = max(len(r) for r in rows[:header_row_count])
    merged: List[str] = []
    for col_idx in range(width):
        parts = [
            rows[r][col_idx]
            for r in range(header_row_count)
            if col_idx < len(rows[r]) and rows[r][col_idx]
        ]
        merged.append(" ".join(parts) if parts else f"Column {col_idx + 1}")
    return merged, rows[header_row_count:]


def _table_to_markdown(rows: List[List[str | None]]) -> str:
    """Convert a PyMuPDF table (list of rows) into a compact markdown table.

    Detects and merges multi-row headers so each column gets a compound label
    like 'Quarter Ended 30-09-2025 (Unaudited)' rather than just 'Quarter Ended'.
    """
    cleaned = [
        [(c or "").strip().replace("\n", " ") for c in row]
        for row in rows
        if row and any((c or "").strip() for c in row)
    ]
    if not cleaned:
        return ""
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    header, body = _merge_multirow_header(cleaned)
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for r in body:
        md += "| " + " | ".join(r) + " |\n"
    return md.strip()


def _extract_images(doc: fitz.Document, page: fitz.Page, page_no: int) -> List[ImageRef]:
    """Extract embedded images from a page, filtering out junk."""
    from . import vision  # local import to avoid cycle / cost on import
    out: List[ImageRef] = []
    seen_xrefs: set[int] = set()
    try:
        for idx, info in enumerate(page.get_images(full=True)):
            xref = info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                pix = doc.extract_image(xref)
            except Exception:
                continue
            data = pix.get("image")
            ext  = pix.get("ext") or "png"
            w    = pix.get("width") or 0
            h    = pix.get("height") or 0
            if not data:
                continue
            if not vision.is_meaningful_image(w, h, len(data)):
                continue
            out.append(ImageRef(page=page_no, index=len(out),
                                bytes=data, ext=ext, width=w, height=h))
    except Exception:
        # PDFs with weird image streams can throw; not fatal.
        pass
    return out


def extract_pdf(pdf_path: str | Path, *, with_images: bool = True, doc_id: str = "") -> List[Page]:
    """Extract page-by-page text, tables, and (optionally) images from a PDF."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    pages: List[Page] = []
    doc = fitz.open(pdf_path)
    current_section = ""
    try:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text", sort=True) or ""
            text = _clean(text)
            layout = _extract_layout(page)
            current_section = _guess_section(text, current_section)

            tables: List[str] = []
            try:
                tabs = page.find_tables()
                for t in tabs:
                    try:
                        rows = t.extract()
                    except Exception:
                        continue
                    md = _table_to_markdown(rows)
                    if md and len(md) > 30:
                        tables.append(md)
            except Exception:
                pass

            # --- Bounding-box table reconstruction fallback ---
            # Financial statement pages (P&L, Balance Sheet) often have
            # complex multi-column layouts that neither OpenDataLoader nor
            # PyMuPDF's find_tables() can detect.  When that happens the
            # numbers end up as flat text and lose their row-column
            # associations, producing wrong answers.  Use the spatial
            # positions of text blocks to reconstruct the table.
            if is_financial_results_page(text):
                try:
                    page_dict = page.get_text("dict", sort=True)
                    reconstructed = reconstruct_financial_table(page_dict, text)
                    if reconstructed and len(reconstructed) > 100:
                        # Replace any garbled PyMuPDF tables with the
                        # reconstructed one, which has correct column
                        # alignment.
                        tables = [reconstructed]
                except Exception:
                    pass  # keep whatever PyMuPDF found

            images = _extract_images(doc, page, i) if with_images else []
            pages.append(Page(
                page=i,
                text=text,
                tables=tables,
                images=images,
                doc_id=doc_id,
                section=current_section,
                layout=layout,
            ))
    finally:
        doc.close()
    return pages
