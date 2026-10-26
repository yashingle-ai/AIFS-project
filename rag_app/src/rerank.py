"""Cross-encoder reranking, with an explicit and observable fallback.

Two-stage retrieval: hybrid search pulls a wide candidate set cheaply, then a
cross-encoder re-scores just those candidates. A bi-encoder embeds query and
passage separately so documents can be precomputed; a cross-encoder runs both
through the model together and is far more accurate, but far too slow to apply
to a whole corpus. Recall from stage one, precision from stage two.

STATUS (August 2026): NVIDIA retired the hosted reranking endpoint on
ai.api.nvidia.com. It returns HTTP 410 Gone with an end-of-life date of
2026-05-18, and no reranking model is exposed on integrate.api.nvidia.com
either. RERANK_BACKEND therefore defaults to "none" and the pipeline runs on
hybrid fusion alone.

Why this module no longer swallows failures:
    The previous version caught every exception and returned the original
    ordering. That is the correct *behaviour* — a reranker outage must never
    break a query — but it was invisible. A dead endpoint looked identical to a
    successful rerank, and /api/info kept reporting rerank_enabled=true. It had
    been failing silently for roughly three months before anyone noticed.

    Now a failure is logged once, counted in rag_rerank_total, and trips a
    circuit breaker so a retired endpoint is not called again for the life of
    the process. is_enabled() and status() then report the truth, so callers
    fall back to hybrid fusion explicitly rather than by accident.

Env:
    RERANK_BACKEND=none     (default) hybrid fusion only
    RERANK_BACKEND=nvidia   legacy hosted endpoint — currently EOL, see above
"""
from __future__ import annotations

import os
from typing import List

import httpx

from .observability import log, rerank_total


_NVIDIA_RERANK_URL = (
    "https://ai.api.nvidia.com/v1/retrieval/"
    "nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking"
)
_RERANK_MODEL = "nvidia/llama-3.2-nv-rerankqa-1b-v2"

# Circuit breaker: set once the backend proves permanently unusable, so we stop
# paying a network timeout on every query for an endpoint that will never answer.
_unavailable_reason: str | None = None


def active_backend() -> str:
    """Configured backend from RERANK_BACKEND. Defaults to 'none' (see module doc)."""
    return os.getenv("RERANK_BACKEND", "none").strip().lower()


def is_enabled() -> bool:
    """Whether reranking will actually run.

    False when disabled by config OR when the backend has already failed in this
    process. Callers use this to decide whether they are reranking or falling
    back, so it must reflect reality rather than intent.
    """
    return active_backend() != "none" and _unavailable_reason is None


def status() -> str:
    """'disabled' | 'active' | 'unavailable' — surfaced by /api/info."""
    if active_backend() == "none":
        return "disabled"
    return "unavailable" if _unavailable_reason else "active"


def unavailable_reason() -> str | None:
    """Why reranking is off, when it failed rather than being disabled."""
    return _unavailable_reason


def _trip(reason: str) -> None:
    """Record a permanent failure and stop calling the backend."""
    global _unavailable_reason
    if _unavailable_reason is None:          # log once, not once per query
        _unavailable_reason = reason
        log.warning(
            "reranker unavailable - falling back to hybrid fusion for all "
            "subsequent queries",
            extra={"rerank_backend": active_backend(), "reason": reason},
        )


def _nvidia_rerank(query: str, passages: List[str], timeout: float = 20.0) -> List[dict]:
    """Call the hosted reranker. Returns [{index, logit}, ...].

    Raises RuntimeError on a retired endpoint (404/410) so the caller can tell a
    permanent failure apart from a transient one.
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set")
    r = httpx.post(
        _NVIDIA_RERANK_URL,
        headers={"Authorization": f"Bearer {api_key}", "accept": "application/json"},
        json={
            "model": _RERANK_MODEL,
            "query": {"text": query},
            "passages": [{"text": p[:4000]} for p in passages],  # NIM payload cap
        },
        timeout=timeout,
    )
    if r.status_code in (404, 410):
        raise RuntimeError(f"endpoint retired (HTTP {r.status_code}): {r.text[:120]}")
    r.raise_for_status()
    return r.json().get("rankings", [])


def rerank(query: str, hits: list, top_k: int) -> list:
    """Re-score hits with a cross-encoder and return the best top_k.

    Never raises. On any failure the hybrid-fusion ordering is returned unchanged
    and the reason is logged and counted — callers still get usable results, but
    the degradation is recorded instead of hidden.

    rerank_score is set on the hits only when reranking actually ran, so
    downstream code (hierarchical._global_score, the API response) can tell
    reranked results apart from fused ones.
    """
    if not hits:
        return hits

    if active_backend() == "none":
        rerank_total.labels("disabled").inc()
        return hits[:top_k]

    if _unavailable_reason is not None:
        rerank_total.labels("unavailable").inc()
        return hits[:top_k]

    if len(hits) <= 1:
        rerank_total.labels("disabled").inc()
        return hits[:top_k]

    try:
        ranks = _nvidia_rerank(query, [h.chunk.text for h in hits])
    except RuntimeError as exc:
        # Permanent: retired endpoint or missing credentials. Stop calling it.
        _trip(str(exc))
        rerank_total.labels("unavailable").inc()
        return hits[:top_k]
    except Exception as exc:
        # Transient: network blip, timeout, rate limit. Keep the backend enabled.
        log.warning("rerank call failed, using hybrid fusion order",
                    extra={"error": f"{type(exc).__name__}: {exc}"[:200]})
        rerank_total.labels("failed").inc()
        return hits[:top_k]

    scores = {r["index"]: float(r["logit"]) for r in ranks}
    for i, h in enumerate(hits):
        setattr(h, "rerank_score", scores.get(i, 0.0))
    rerank_total.labels("active").inc()
    return sorted(hits, key=lambda h: getattr(h, "rerank_score", 0.0), reverse=True)[:top_k]
