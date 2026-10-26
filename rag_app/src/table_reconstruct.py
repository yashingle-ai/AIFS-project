"""Bounding-box-based financial table reconstruction.

When OpenDataLoader and PyMuPDF's ``find_tables()`` both fail to detect a
financial statement table (common with complex multi-column P&L / balance-sheet
layouts), this module reconstructs the table from the **spatial positions** of
text blocks extracted via PyMuPDF's ``get_text("dict")``.

Algorithm
---------
1. Detect financial-statement pages via header text.
2. Extract all text spans with bounding boxes.
3. Group spans into rows by Y-coordinate proximity.
4. Identify column boundaries from date-header X-positions (e.g. ``30-09-2025``).
5. For each row, pair the leftmost label text with values at each column X.
6. Emit a clean markdown pipe-table with unambiguous column headers.

The resulting table flows through the existing enrichment pipeline
(``extract_table_relationships``) so every value is correctly bound to its
row label AND column period.

1. Detect financial-statement page        
2. Extract text + bounding boxes     
3. Group text into rows using Y      
4. Find columns using date X positions       
5. Match labels with values       
6. Generate Markdown table
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

# Detection patterns
_FINANCIAL_RESULTS_RE = re.compile(
    r"(?:STATEMENT\s+OF\s+)?(?:UN)?AUDITED\s+"
    r"(?:CONSOLIDATED|STANDALONE)?\s*"
    r"FINANCIAL\s+RESULTS",
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")
_QUALIFIER_RE = re.compile(r"^\((?:Un)?[Aa]udited\)$")
_PERIOD_RE = re.compile(r"^(?:Quarter|Half\s*Year|Year)\s+Ended$", re.IGNORECASE)
_VALUE_RE = re.compile(r"^[\(\-\u2212\u2013\u2014]?\d[\d,]*(?:\.\d+)?[\)]*$")
_SR_NO_RE = re.compile(r"^\d{1,2}$")
_SKIP_RE = re.compile(
    r"^(?:adani|phone|fax|cin|registered|website|email|highway|ahmedabad|"
    r"shantigram|vaishno|corporate\s+house)",
    re.IGNORECASE,
)
# Stop markers – rows that indicate we've gone past the table body.
# Be careful: some data rows contain parenthetical note references like
# "(Refer Note 3 & 4)" which should NOT trigger a stop.
_STOP_RE = re.compile(
    r"(?:^Additional\s+Disclosure|^Notes:\s|EPS\s+has\s+been|"
    r"See\s+accompanying|for\s+and\s+on\s+behalf|Chartered\s+Accountant)",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(r"in\s+Crores", re.IGNORECASE)


@dataclass
class _Span:
    """One positioned text run: its string plus the bounding box PyMuPDF reported."""
    text: str
    x0: float
    y0: float
    x1: float
    y1: float


def is_financial_results_page(page_text: str) -> bool:
    """Return True if the page contains a financial-results statement header."""
    return bool(_FINANCIAL_RESULTS_RE.search(page_text))


def reconstruct_financial_table(
    page_dict: dict,
    page_text: str,
) -> Optional[str]:
    """Reconstruct a financial-statement table from bounding-box positions.

    Parameters
    ----------
    page_dict : dict
        Output of ``fitz.Page.get_text("dict", sort=True)``.
    page_text : str
        Plain text of the page (``fitz.Page.get_text("text")``).

    Returns
    -------
    str | None
        Markdown pipe-table, or ``None`` if reconstruction is not applicable
        or the page doesn't look like a financial-results statement.
    """
    if not is_financial_results_page(page_text):
        return None

    spans = _extract_spans(page_dict)
    if len(spans) < 20:
        return None

    rows = _group_by_rows(spans, tolerance=3.0)
    if len(rows) < 10:
        return None

    # --- Phase 1: locate column positions from date headers ---
    date_spans: List[_Span] = []
    period_spans: List[_Span] = []
    qualifier_spans: List[_Span] = []

    for span in spans:
        if _DATE_RE.match(span.text):
            date_spans.append(span)
        elif _PERIOD_RE.match(span.text):
            period_spans.append(span)
        elif _QUALIFIER_RE.match(span.text):
            qualifier_spans.append(span)

    if len(date_spans) < 2:
        return None

    # Bucket dates by X-position (tolerance 8px)
    x_buckets = _cluster_x_positions([d.x0 for d in date_spans], tolerance=8.0)
    col_x = sorted(x_buckets.keys())

    # Build column metadata
    col_info: dict[float, dict] = {x: {"dates": [], "period": "", "qualifier": ""} for x in col_x}
    for d in date_spans:
        bucket = _nearest_bucket(d.x0, col_x)
        col_info[bucket]["dates"].append(d.text)

    # Assign period labels to closest column group
    for p in period_spans:
        bucket = _nearest_bucket(p.x0, col_x)
        if bucket is not None:
            col_info[bucket]["period"] = p.text

    # Assign qualifiers
    for q in qualifier_spans:
        bucket = _nearest_bucket(q.x0, col_x)
        if bucket is not None:
            col_info[bucket]["qualifier"] = q.text

    # Propagate period labels rightward (e.g. "Quarter Ended" covers the next 2-3 cols)
    _propagate_periods(col_x, col_info, period_spans)

    label_x_max = col_x[0] - 15  # labels are to the left of the first value column

    # --- Phase 2: build column headers ---
    headers = ["Particulars"]
    for x in col_x:
        info = col_info[x]
        parts = []
        if info["period"]:
            parts.append(info["period"])
        if info["dates"]:
            parts.append(info["dates"][0])
        if info["qualifier"]:
            parts.append(info["qualifier"])
        headers.append(" ".join(parts) if parts else "Column")

    # --- Phase 3: find the first data row ---
    data_start_y = _find_data_start(rows, label_x_max, col_x)
    if data_start_y is None:
        return None

    # --- Phase 4: extract data rows ---
    data_rows: List[List[str]] = []
    for row in rows:
        if row[0].y0 < data_start_y - 5:
            continue

        label_parts: List[str] = []
        for span in row:
            if span.x0 >= label_x_max:
                break
            if _SR_NO_RE.match(span.text) and span.x0 < 45:
                continue
            if not _is_numeric(span.text):
                label_parts.append(span.text)

        label = " ".join(label_parts).strip()
        if not label:
            continue
        if _SKIP_RE.match(label):
            continue
        if _STOP_RE.search(label):
            break

        # Assign value spans to columns
        values = [""] * len(col_x)
        for span in row:
            if not _is_numeric(span.text):
                continue
            if span.x0 < label_x_max - 20:
                continue  # probably a Sr.No. digit in the label column
            bucket = _nearest_bucket(span.x0, col_x, max_dist=30)
            if bucket is not None:
                idx = col_x.index(bucket)
                values[idx] = span.text

        data_rows.append([label] + values)

    if len(data_rows) < 5:
        return None

    # --- Phase 5: format markdown table ---
    lines = [
        "| " + " | ".join(_escape(c) for c in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for dr in data_rows:
        while len(dr) < len(headers):
            dr.append("")
        lines.append("| " + " | ".join(_escape(c) for c in dr) + " |")

    return "\n".join(lines)


def _extract_spans(page_dict: dict) -> List[_Span]:
    """Flatten a PyMuPDF page dict into positioned spans."""
    out: List[_Span] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            parts: List[str] = []
            x0, y0, x1, y1 = 1e9, 1e9, 0.0, 0.0
            for span in line.get("spans", []):
                t = span.get("text", "")
                if t.strip():
                    parts.append(t)
                    b = span.get("bbox", [0, 0, 0, 0])
                    x0 = min(x0, b[0])
                    y0 = min(y0, b[1])
                    x1 = max(x1, b[2])
                    y1 = max(y1, b[3])
            text = "".join(parts).strip()
            if text and x0 < 1e9:
                out.append(_Span(text=text, x0=round(x0, 1), y0=round(y0, 1),
                                 x1=round(x1, 1), y1=round(y1, 1)))
    return out


def _is_numeric(text: str) -> bool:
    """Whether a span reads as a financial figure rather than a label."""
    cleaned = text.replace(" ", "").replace("\u2212", "-").replace("\u2013", "-")
    return bool(_VALUE_RE.match(cleaned))


def _group_by_rows(spans: List[_Span], tolerance: float) -> List[List[_Span]]:
    """Group spans into visual rows by vertical proximity.

    Cells on one printed row share a baseline within a small tolerance, which
    is what makes rows recoverable once the PDF's own table structure is gone.
    """
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda s: (s.y0, s.x0))
    rows: List[List[_Span]] = []
    cur: List[_Span] = [sorted_spans[0]]
    cur_y = sorted_spans[0].y0
    for span in sorted_spans[1:]:
        if abs(span.y0 - cur_y) <= tolerance:
            cur.append(span)
        else:
            rows.append(sorted(cur, key=lambda s: s.x0))
            cur = [span]
            cur_y = span.y0
    if cur:
        rows.append(sorted(cur, key=lambda s: s.x0))
    return rows


def _cluster_x_positions(xs: List[float], tolerance: float) -> dict[float, List[float]]:
    """Cluster a list of X-coords into buckets within *tolerance*."""
    sorted_xs = sorted(xs)
    buckets: dict[float, List[float]] = {}
    for x in sorted_xs:
        matched = False
        for bk in buckets:
            if abs(x - bk) <= tolerance:
                buckets[bk].append(x)
                matched = True
                break
        if not matched:
            buckets[x] = [x]
    return buckets


def _nearest_bucket(x: float, bucket_keys: List[float], max_dist: float = 30) -> Optional[float]:
    """Snap an x-position to the closest column, or None if nothing is near enough."""
    best, best_dist = None, max_dist + 1
    for bk in bucket_keys:
        d = abs(x - bk)
        if d < best_dist:
            best, best_dist = bk, d
    return best if best_dist <= max_dist else None


def _propagate_periods(
    col_x: List[float],
    col_info: dict[float, dict],
    period_spans: List[_Span],
) -> None:
    """Assign period labels to date columns using spatial grouping.

    Strategy: period spans (e.g. "Quarter Ended") sit ABOVE a group of date
    columns.  We assign them by checking which date columns are nearby on the
    X-axis.  The key insight is that the *first* date column in a group is
    sometimes to the LEFT of the period label (because the Particulars column
    is on the far left), so we extend the propagation range leftward by one
    column gap.
    """
    if not col_x or not period_spans:
        return

    # Sort periods by X position
    sorted_periods = sorted(period_spans, key=lambda p: p.x0)
    col_gap = min(
        (col_x[i + 1] - col_x[i] for i in range(len(col_x) - 1)),
        default=60.0,
    )

    for p in sorted_periods:
        p_width = p.x1 - p.x0
        # Extend range: leftward by one column gap, rightward by the span width + gap
        left_bound = p.x0 - col_gap - 10
        right_bound = p.x0 + p_width + col_gap
        for x in col_x:
            if left_bound <= x <= right_bound:
                if not col_info[x]["period"]:
                    col_info[x]["period"] = p.text


def _find_data_start(
    rows: List[List[_Span]],
    label_x_max: float,
    col_x: List[float],
) -> Optional[float]:
    """Return the Y-coordinate of the first row that has a label AND values."""
    for row in rows:
        label_spans = [
            s for s in row
            if s.x0 < label_x_max
            and not _is_numeric(s.text)
            and not _SKIP_RE.match(s.text)
            and not _SR_NO_RE.match(s.text)
            and not _DATE_RE.match(s.text)
            and not _QUALIFIER_RE.match(s.text)
            and not _PERIOD_RE.match(s.text)
            and not _CURRENCY_RE.search(s.text)
        ]
        value_spans = [
            s for s in row
            if _is_numeric(s.text) and s.x0 >= label_x_max - 20
        ]
        if label_spans and value_spans:
            return row[0].y0
    return None


def _escape(text: str) -> str:
    """Escape a cell so it cannot break the markdown table grid."""
    return text.replace("|", "\\|")
