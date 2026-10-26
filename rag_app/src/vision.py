"""Vision module — convert PDF images into searchable text descriptions.

Image content in PDFs (charts, diagrams, photos, hand-drawn flows) is invisible
to a text-only retrieval pipeline. This module captions each image with a
vision LLM at INDEX time so the description becomes a regular searchable chunk
indistinguishable from real text. Cost is paid once during indexing, not per
query.

Backends:
  VISION_BACKEND=nvidia (default) → meta/llama-3.2-11b-vision-instruct
  VISION_BACKEND=gemini            → gemini-2.5-flash (multimodal)
  VISION_BACKEND=none              → image captioning disabled

Filtering:
  We skip tiny logos, page-header decorations, very-low-aspect-ratio bars, etc.
  See `is_meaningful_image` for the heuristics.
"""
from __future__ import annotations

import base64
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional


def active_backend() -> str:
    """Vision provider from VISION_BACKEND ('nvidia' default, 'none' disables)."""
    return os.getenv("VISION_BACKEND", "nvidia").strip().lower()


def is_enabled() -> bool:
    """Whether images should be captioned during indexing.

    Two independent switches: VISION_BACKEND=none turns the feature off, and
    INDEX_IMAGES=false keeps the backend configured but skips the work — useful
    for a fast re-index when only the text pipeline changed.
    """
    if active_backend() == "none":
        return False
    if os.getenv("INDEX_IMAGES", "true").strip().lower() in ("0", "false", "no"):
        return False
    return True


# Image filtering: skip junk so we don't spend vision-LLM tokens on it

def is_meaningful_image(width: int, height: int, n_bytes: int) -> bool:
    """Filter out logos, decorative bars, page furniture, pixel-thin separators."""
    if width < 80 or height < 80:
        return False  # tiny → likely icon/logo
    aspect = max(width, height) / max(1, min(width, height))
    if aspect > 12:
        return False  # ribbon / divider / page header bar
    if width * height < 8_000:
        return False  # too small to carry meaningful info
    if n_bytes < 1500:
        return False  # almost certainly a glyph or single-color fill
    return True


# Caption prompt — keep it tight so descriptions are short, factual, indexable

CAPTION_PROMPT = (
    "You are a precise data extraction system for financial document images.\n\n"
    "Extract ALL data from this image into structured format.\n\n"
    "Rules:\n"
    "1. CHART/GRAPH: Extract EVERY visible data point into a markdown table.\n"
    "   Output format:\n"
    "   CHART TYPE: [bar/line/pie/waterfall/stacked bar/area/etc.]\n"
    "   TITLE: [chart title if visible]\n\n"
    "   | Category | Metric1 | Metric2 | ... |\n"
    "   |----------|---------|---------|-----|\n"
    "   | Label1   | value1  | value2  | ... |\n\n"
    "   KEY TREND: [one sentence on the main trend]\n\n"
    "   - Copy numbers EXACTLY as shown (include units: Cr, %, Rs, $, mn, bn)\n"
    "   - Include ALL series/categories from the legend\n"
    "   - If values are read from bar heights, state approximate values\n"
    "   - Include axis labels and units\n\n"
    "2. TABLE: Reproduce as a full markdown table preserving all rows/columns.\n"
    "3. DIAGRAM/FLOWCHART: List entities and relationships as bullet points.\n"
    "4. PHOTO/ILLUSTRATION: Describe what is shown in 1-2 sentences.\n"
    "5. NO CONTENT (decorative banner, blank, watermark): Reply EXACTLY: SKIP\n\n"
    "Do NOT invent data. Do NOT approximate when exact values are visible. "
    "Be exhaustive — extract every number, label, and data point."
)


# Backend dispatch

def _nvidia_describe(image_bytes: bytes, mime: str = "image/png") -> Optional[str]:
    """Caption one image via a NIM vision model.

    The image is inlined as a base64 data URL rather than uploaded, so no
    intermediate storage is involved. Returns None when the model replies SKIP
    or every retry fails.
    """
    from openai import OpenAI
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set")
    base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    model = os.getenv("NVIDIA_VISION_MODEL", "meta/llama-3.2-11b-vision-instruct")
    client = OpenAI(api_key=api_key, base_url=base_url)
    b64 = base64.b64encode(image_bytes).decode()

    last: Exception | None = None
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CAPTION_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                }],
                max_tokens=350,
                temperature=0.0,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            last = e
            msg = str(e)
            if "429" in msg or "rate" in msg.lower():
                time.sleep(min(2 ** attempt * 5, 30))
            else:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"NVIDIA vision failed: {last}")


def _gemini_describe(image_bytes: bytes, mime: str = "image/png") -> Optional[str]:
    """Caption one image via Gemini multimodal. Returns None on SKIP or failure."""
    from google import genai
    from google.genai import types
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    model = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
    client = genai.Client(api_key=api_key)

    last: Exception | None = None
    for attempt in range(3):
        try:
            r = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime),
                    CAPTION_PROMPT,
                ],
                config=types.GenerateContentConfig(temperature=0.0),
            )
            return (r.text or "").strip()
        except Exception as e:
            last = e
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"Gemini vision failed: {last}")


def describe_image(image_bytes: bytes, mime: str = "image/png") -> Optional[str]:
    """Return a textual description of the image, or None if SKIPPED / failed."""
    if not is_enabled():
        return None
    backend = active_backend()
    try:
        if backend == "nvidia":
            text = _nvidia_describe(image_bytes, mime)
        elif backend == "gemini":
            text = _gemini_describe(image_bytes, mime)
        else:
            return None
    except Exception:
        # Vision failures must NEVER break indexing.
        return None
    if not text:
        return None
    if text.strip().upper().startswith("SKIP"):
        return None
    return text


# Concurrent batch helper — captioning N images is the slowest indexing step

def describe_batch(images: list[dict], max_workers: int = 4,
                   on_progress=None) -> list[Optional[str]]:
    """images: [{ 'bytes': b'...', 'mime': 'image/png' }, ...]  (order preserved).
    Returns parallel list of descriptions (None for skipped/failed)."""
    if not images or not is_enabled():
        return [None] * len(images)

    out: list[Optional[str]] = [None] * len(images)
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(describe_image, im["bytes"], im.get("mime", "image/png")): i
            for i, im in enumerate(images)
        }
        for f in as_completed(futures):
            i = futures[f]
            try:
                out[i] = f.result()
            except Exception:
                out[i] = None
            done += 1
            if on_progress:
                try: on_progress(done, len(images))
                except Exception: pass
    return out
