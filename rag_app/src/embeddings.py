"""Embeddings — dispatches to NVIDIA NIM or Google Gemini based on EMBED_BACKEND.

Public API:
  - embed_documents(texts) -> (N, D) float32, L2-normalized
  - embed_query(text)      -> (1, D) float32, L2-normalized
  - active_backend()       -> "nvidia" | "gemini"
  - active_model()         -> model id (string)
  - active_dim()           -> embedding dimension (int) — known after first call

Why L2-normalize here:
  retriever.py indexes with faiss.IndexFlatIP, which scores by inner product.
  For unit vectors inner product IS cosine similarity, so normalizing at this
  boundary is what makes Hit.dense_score a bounded [-1, 1] number. The refusal
  floor compares against a fixed threshold and only works because of that.

Why documents and queries are embedded differently:
  Passages go out as input_type="passage" (NIM) / RETRIEVAL_DOCUMENT (Gemini),
  queries as "query" / RETRIEVAL_QUERY. The E5 family is trained asymmetrically;
  encoding both sides identically still "works" but quietly costs recall, which
  makes it an expensive mistake to debug later.

Rate handling:
  Batch sizes track each provider's limit (NIM 32, Gemini 50). _pace() enforces
  a minimum gap between calls and _maybe_429_wait() parses the retryDelay hint
  out of the error body, so indexing a large PDF backs off instead of failing.
"""
from __future__ import annotations

import os
import re
import time
from collections import OrderedDict
from typing import List, Sequence

import numpy as np

from . import observability as _obs


_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s")
_last_call_at: float = 0.0
_known_dim: int | None = None


def active_backend() -> str:
    """Embedding provider from EMBED_BACKEND ('nvidia' default)."""
    return os.getenv("EMBED_BACKEND", "nvidia").strip().lower()


def active_model() -> str:
    """Model id for the active backend, overridable per-provider via env."""
    if active_backend() == "nvidia":
        return os.getenv("NVIDIA_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
    return os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")


def active_dim() -> int | None:
    """Embedding width, or None before the first call.

    Learned from the first response rather than hardcoded, so switching model
    or provider does not require editing a constant.
    """
    return _known_dim


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, in float32.

    Zero-norm rows would divide by zero, so their norm is forced to 1 — they
    stay all-zero and simply never match anything, which is the desired
    behaviour for an empty or unembeddable chunk.
    """
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32, copy=False)


def _pace(min_gap: float) -> None:
    """Sleep so consecutive API calls are at least `min_gap` apart.

    Prevents a large document's embedding loop from tripping provider rate
    limits. NOTE: `_last_call_at` is a module global with no lock, so this does
    not hold across concurrent indexing threads.
    """
    global _last_call_at
    now = time.monotonic()
    delta = now - _last_call_at
    if delta < min_gap:
        time.sleep(min_gap - delta)
    _last_call_at = time.monotonic()


def _maybe_429_wait(msg: str) -> bool:
    """Sleep and report True if `msg` looks like a rate-limit error.

    Providers put the retry hint in the error body rather than a header, so the
    delay is parsed out of the text when present and otherwise defaults to 30s.
    """
    if "429" not in msg and "RESOURCE_EXHAUSTED" not in msg and "rate" not in msg.lower():
        return False
    m = _RETRY_DELAY_RE.search(msg)
    wait = float(m.group(1)) if m else 30.0
    time.sleep(max(wait + 1.0, 5.0))
    return True


_NVIDIA_BATCH = 32      # NIM embeddings cap is small; 32 is a safe sweet spot
_NVIDIA_MIN_GAP = 0.1


_nvidia_clients: dict = {}


def _nvidia_client():
    """OpenAI-compatible client pointed at NVIDIA NIM. Raises if the key is unset.

    Cached per (key, base_url). Constructing a new OpenAI() builds a fresh
    connection pool, so a new TLS handshake was being paid on every single
    call — measured at ~5.2s cold versus ~1.0s on a warm connection. Reusing
    the client lets HTTP keep-alive do its job.
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set. Get one at https://build.nvidia.com")
    base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    cached = _nvidia_clients.get((api_key, base_url))
    if cached is None:
        from openai import OpenAI
        cached = OpenAI(api_key=api_key, base_url=base_url)
        _nvidia_clients[(api_key, base_url)] = cached
    return cached


def _nvidia_embed(texts: Sequence[str], input_type: str) -> np.ndarray:
    """input_type: 'passage' for documents, 'query' for queries."""
    global _known_dim
    client = _nvidia_client()
    model = active_model()
    out: List[List[float]] = []
    total = len(texts)
    for i in range(0, total, _NVIDIA_BATCH):
        batch = list(texts[i:i + _NVIDIA_BATCH])
        last_err: Exception | None = None
        for attempt in range(4):
            _pace(_NVIDIA_MIN_GAP)
            # counted here, not at the public API, so the metric reflects real
            # network traffic including retries rather than caller intent
            _obs.embed_api_calls_total.labels(
                "nvidia", "query" if input_type == "query" else "passage").inc()
            try:
                resp = client.embeddings.create(
                    model=model,
                    input=batch,
                    encoding_format="float",
                    extra_body={"input_type": input_type, "truncate": "END"},
                )
                out.extend(d.embedding for d in resp.data)
                break
            except Exception as e:
                last_err = e
                if _maybe_429_wait(str(e)):
                    continue
                time.sleep(min(2 ** attempt, 10))
        else:
            raise RuntimeError(f"NVIDIA embed failed: {last_err}")

        if total > _NVIDIA_BATCH:
            done = min(i + _NVIDIA_BATCH, total)
            print(f"  embed {done}/{total}", flush=True)

    arr = np.asarray(out, dtype=np.float32)
    _known_dim = arr.shape[1] if arr.size else _known_dim
    return _normalize(arr)


_GEMINI_BATCH = 50
_GEMINI_MIN_GAP = 0.3


def _gemini_client():
    """Google GenAI client. Raises if GEMINI_API_KEY is unset."""
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Get one at https://aistudio.google.com/apikey")
    return genai.Client(api_key=api_key)


def _gemini_embed(texts: Sequence[str], task_type: str) -> np.ndarray:
    """Embed via Gemini in batches, retrying on rate limits.

    task_type selects the asymmetric mode: RETRIEVAL_DOCUMENT for passages,
    RETRIEVAL_QUERY for queries.
    """
    global _known_dim
    from google.genai import types
    client = _gemini_client()
    model = active_model()
    out: List[List[float]] = []
    total = len(texts)
    for i in range(0, total, _GEMINI_BATCH):
        batch = list(texts[i:i + _GEMINI_BATCH])
        last_err: Exception | None = None
        for attempt in range(4):
            _pace(_GEMINI_MIN_GAP)
            _obs.embed_api_calls_total.labels(
                "gemini", "query" if task_type == "RETRIEVAL_QUERY" else "passage").inc()
            try:
                resp = client.models.embed_content(
                    model=model,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                out.extend(e.values for e in resp.embeddings)
                break
            except Exception as e:
                last_err = e
                if _maybe_429_wait(str(e)):
                    continue
                time.sleep(min(2 ** attempt, 10))
        else:
            raise RuntimeError(f"Gemini embed failed: {last_err}")

        if total > _GEMINI_BATCH:
            done = min(i + _GEMINI_BATCH, total)
            print(f"  embed {done}/{total}", flush=True)

    arr = np.asarray(out, dtype=np.float32)
    _known_dim = arr.shape[1] if arr.size else _known_dim
    return _normalize(arr)


def embed_documents(texts: Sequence[str]) -> np.ndarray:
    """Embed passages for indexing. Returns (N, D) L2-normalized float32.

    The empty case still has to return a correctly shaped array, so it falls
    back to the last known width — or 1024 if nothing has been embedded yet.
    """
    if not texts:
        return np.zeros((0, _known_dim or 1024), dtype=np.float32)
    backend = active_backend()
    if backend == "nvidia":
        return _nvidia_embed(texts, "passage")
    if backend == "gemini":
        return _gemini_embed(texts, "RETRIEVAL_DOCUMENT")
    raise ValueError(f"Unknown EMBED_BACKEND: {backend}")


_QUERY_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_QUERY_CACHE_MAX = 512


def embed_query(text: str) -> np.ndarray:
    """Embed a single query. Returns (1, D), encoded in query mode.

    Cached, because one user question triggers this many times over:
    HybridRetriever._dense() calls it per index, and a question fans out across
    sub-queries x index tiers x documents. Measured at 6 identical calls for a
    single question with 2 documents indexed, each a ~1s network round trip.

    The cache key includes backend and model so switching providers mid-process
    cannot serve a vector from the wrong embedding space.
    """
    backend = active_backend()
    key = (backend, active_model(), text)

    hit = _QUERY_CACHE.get(key)
    if hit is not None:
        _QUERY_CACHE.move_to_end(key)      # LRU: mark as recently used
        _obs.embed_cache_total.labels("hit").inc()
        return hit
    _obs.embed_cache_total.labels("miss").inc()

    if backend == "nvidia":
        vec = _nvidia_embed([text], "query")
    elif backend == "gemini":
        vec = _gemini_embed([text], "RETRIEVAL_QUERY")
    else:
        raise ValueError(f"Unknown EMBED_BACKEND: {backend}")

    _QUERY_CACHE[key] = vec
    if len(_QUERY_CACHE) > _QUERY_CACHE_MAX:
        _QUERY_CACHE.popitem(last=False)   # evict least recently used
    return vec


def clear_query_cache() -> None:
    """Drop cached query vectors. Only needed in tests or after a model swap."""
    _QUERY_CACHE.clear()
