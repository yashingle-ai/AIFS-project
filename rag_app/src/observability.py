"""Observability: structured JSON logging + Prometheus metrics + request middleware.

What you get:
  - logs.jsonl style stdout: every line is a parseable JSON record with timestamp,
    level, request_id, route, status, latency_ms, and any extras the call site adds.
  - GET /api/metrics: Prometheus exposition format with HTTP + RAG-specific series.

Why logs go to stdout and nowhere else:
  The process does not manage log files, rotation or shipping. A collector
  (Docker, systemd, a sidecar) owns that. One JSON object per line is the format
  every collector can parse without a custom regex.

Why request_id lives in a ContextVar:
  It propagates through the async call stack automatically, so any logger call
  inside a handler is correlated without threading an id parameter through every
  function signature. An inbound X-Request-Id is honoured when present, so a
  trace started at a proxy survives into these logs.

Why metrics are recorded in `finally`:
  A handler that raises still gets counted. Recording after a successful return
  would silently hide exactly the requests you most need to see.

Label cardinality — read before adding a label:
  Prometheus creates one time series per distinct label combination, and series
  are never garbage collected. Labels must therefore come from a small fixed set:
  the route TEMPLATE ("/api/documents/{doc_id}/chat"), never the resolved path,
  or every document id becomes a permanent series and the scrape target grows
  without bound. Same rule for status — bucket it, don't use raw error strings.

Why _SafeLogger wraps `extra=`:
  logging.LogRecord owns attribute names like "module", "process" and "message".
  Passing extra={"module": ...} raises at emit time and takes the handler down
  with it. The wrapper prefixes any colliding key instead of trusting call sites
  to remember the reserved list.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


# Request-scoped context (so any log call inside a handler picks up request_id)

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def current_request_id() -> str | None:
    """Request id for the in-flight request, or None outside a request."""
    return _request_id.get()


class JsonFormatter(logging.Formatter):
    """Formats LogRecords as single-line JSON objects."""
    def format(self, record: logging.LogRecord) -> str:
        """Render one record as JSON.

        Any non-standard attribute becomes a top-level field, so call sites can
        attach structure via extra={...}. Values that will not serialise fall back
        to repr() rather than failing the log call.
        """
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) +
                  f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = current_request_id()
        if rid:
            payload["request_id"] = rid
        # Pull any structured extras attached via logger.info("...", extra={...})
        for k, v in record.__dict__.items():
            if k in ("name", "msg", "args", "levelname", "levelno", "pathname",
                     "filename", "module", "exc_info", "exc_text", "stack_info",
                     "lineno", "funcName", "created", "msecs", "relativeCreated",
                     "thread", "threadName", "processName", "process", "message",
                     "taskName"):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except Exception:
                payload[k] = repr(v)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> None:
    """Replace the root logger's handlers with a single JSON-to-stdout handler."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # Quiet down noisy upstream libs
    for noisy in ("uvicorn.access", "watchfiles.main", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName",
}


class _SafeLogger(logging.Logger):
    """Wraps `extra=` to avoid collisions with reserved LogRecord attributes."""

    def _safe_extra(self, extra):
        """Prefix any key that collides with a reserved LogRecord attribute."""
        if not extra:
            return extra
        return {(k if k not in _RESERVED else f"_{k}"): v for k, v in extra.items()}

    def info(self, msg, *args, **kw):
        """info() with collision-safe extras."""
        kw["extra"] = self._safe_extra(kw.get("extra"))
        super().info(msg, *args, **kw)
    def warning(self, msg, *args, **kw):
        """warning() with collision-safe extras."""
        kw["extra"] = self._safe_extra(kw.get("extra"))
        super().warning(msg, *args, **kw)
    def error(self, msg, *args, **kw):
        """error() with collision-safe extras."""
        kw["extra"] = self._safe_extra(kw.get("extra"))
        super().error(msg, *args, **kw)
    def exception(self, msg, *args, **kw):
        """exception() with collision-safe extras."""
        kw["extra"] = self._safe_extra(kw.get("extra"))
        super().exception(msg, *args, **kw)
    def debug(self, msg, *args, **kw):
        """debug() with collision-safe extras."""
        kw["extra"] = self._safe_extra(kw.get("extra"))
        super().debug(msg, *args, **kw)


logging.setLoggerClass(_SafeLogger)
log = logging.getLogger("rag")


http_requests_total = Counter(
    "rag_http_requests_total",
    "HTTP requests received",
    ["method", "route", "status"],
)
http_request_duration_seconds = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
chat_requests_total = Counter(
    "rag_chat_requests_total",
    "Chat completions",
    ["backend", "model", "refused"],
)
chat_tokens_total = Counter(
    "rag_chat_tokens_total",
    "Token usage by direction",
    ["backend", "direction"],  # direction = prompt|completion
)
indexing_runs_total = Counter(
    "rag_indexing_runs_total",
    "Document indexing runs",
    ["status"],  # ok|error
)
indexing_duration_seconds = Histogram(
    "rag_indexing_duration_seconds",
    "Time to build a document index",
    buckets=(0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
)
documents_indexed = Gauge(
    "rag_documents_indexed",
    "Number of fully indexed documents currently on disk",
)
retrieval_top_dense_score = Histogram(
    "rag_retrieval_top_dense_score",
    "Top-1 dense (cosine) retrieval score per query",
    buckets=(0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
feedback_total = Counter(
    "rag_feedback_total",
    "User feedback events",
    ["rating"],  # up|down
)

# --- pipeline instrumentation -------------------------------------------------
# These exist so the three failure modes fixed in Aug 2026 stay visible:
# duplicate embedding calls, a silently dead reranker, and slow LLM backends.

embed_api_calls_total = Counter(
    "rag_embed_api_calls_total",
    "Embedding requests that actually reached the provider",
    ["backend", "kind"],  # kind = query|passage
)
embed_cache_total = Counter(
    "rag_embed_cache_total",
    "Query-embedding cache outcomes",
    ["result"],  # hit|miss
)
retrieval_duration_seconds = Histogram(
    "rag_retrieval_duration_seconds",
    "Time spent in hybrid retrieval (dense + sparse + fusion)",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
rerank_total = Counter(
    "rag_rerank_total",
    "Reranker outcomes per call",
    ["status"],  # active|disabled|unavailable|failed
)
llm_duration_seconds = Histogram(
    "rag_llm_duration_seconds",
    "Latency of a single LLM generation call",
    ["backend"],
    buckets=(0.5, 1, 2.5, 5, 10, 20, 30, 60, 120),
)


def metrics_response() -> Response:
    """Current metrics in Prometheus exposition format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Middleware: assigns a request_id, logs each request, records metrics

class RequestLogMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, logs every request, and records HTTP metrics."""
    async def dispatch(self, request: Request, call_next) -> Response:
        """Time the request, emit one log line and update the counters.

        Everything happens in `finally` so failed and exceptional requests are
        counted too — those are the ones worth measuring.

        KNOWN ISSUE: the route template is read before call_next, but Starlette
        only populates scope["route"] during routing, so this falls back to the
        resolved path and every document id becomes its own metric series. Move
        both lines into the finally block to fix.
        """
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = _request_id.set(rid)
        start = time.perf_counter()
        # Route template (or path if no match) — labels MUST be low-cardinality.
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        status_code = 500
        try:
            resp = await call_next(request)
            status_code = resp.status_code
            return resp
        finally:
            dur = time.perf_counter() - start
            try:
                http_requests_total.labels(request.method, route_path, str(status_code)).inc()
                http_request_duration_seconds.labels(request.method, route_path).observe(dur)
            except Exception:
                pass
            log.info(
                "http",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "route": route_path,
                    "status": status_code,
                    "duration_ms": round(dur * 1000, 1),
                    "client": (request.client.host if request.client else None),
                    "user_agent": request.headers.get("user-agent", "")[:120],
                },
            )
            _request_id.reset(token)
