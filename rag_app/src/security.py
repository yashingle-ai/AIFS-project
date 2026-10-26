"""Security middleware + helpers for the FastAPI server.

Layers:
  1. CORS restricted to localhost (override via ALLOW_ORIGINS env, csv).
  2. Strict response headers (CSP, no-sniff, frame-deny, referrer-policy).
  3. Optional bearer-token auth (set RAG_AUTH_TOKEN to enable).
  4. Per-IP token-bucket rate limiter with separate buckets for upload vs chat.
  5. Path-traversal-safe doc_id validation.
  6. Filename sanitization for uploads.

All defaults are safe-by-default for a single-user demo; override via env.

Threat model this actually covers:
  A document id arrives from the URL and is used to build a filesystem path, and
  a PDF filename arrives from a multipart upload. Both are attacker-controlled
  strings that reach the disk, so both are validated rather than trusted. Chat
  payloads are bounded so a single request cannot blow up the prompt budget.

Why doc_id is checked three separate ways:
  An allowlist regex, an explicit ".." / leading-dot check, and a PurePosixPath
  round-trip. Any one of them would probably do; together they survive someone
  later relaxing the regex without realising what it was protecting. Path
  traversal is cheap to prevent and expensive to discover in production.

Why the bearer comparison uses hmac.compare_digest:
  A plain == returns as soon as two bytes differ, and the timing difference is
  measurable over enough requests. compare_digest runs in constant time.

Why the rate limiter buckets upload separately:
  An upload triggers PDF parsing, vision captioning and a full embedding run —
  seconds of CPU and real API spend. A chat request costs one round trip. One
  shared limit would either throttle chat pointlessly or leave indexing exposed.

Known limitation — X-Forwarded-For:
  _ip() trusts the header when present, which is correct behind a proxy that
  overwrites it and wrong when the app is exposed directly, where a client can
  spoof it to defeat the limiter. Gate this on a TRUST_PROXY flag before running
  without a reverse proxy in front.

Note on the CSP:
  script-src carries 'unsafe-inline' and 'unsafe-eval' because the SPA compiles
  JSX in the browser via Babel from a CDN. That is a real weakening of the
  policy, accepted deliberately to keep the frontend build-free. A bundled
  frontend should drop both and tighten script-src to 'self'.
"""
from __future__ import annotations

import os
import re
import threading
import time
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Tuple

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


def cors_origins() -> list[str]:
    """Allowed origins: ALLOW_ORIGINS csv if set, else the usual dev ports."""
    raw = os.getenv("ALLOW_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    # Safe default: localhost on common dev ports
    return [
        "http://localhost:8000", "http://127.0.0.1:8000",
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:5173", "http://127.0.0.1:5173",
    ]


# CSP allows the CDNs the SPA uses (React, Tailwind, Babel, marked, DOMPurify, fonts).
_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
    "  https://unpkg.com https://cdn.jsdelivr.net https://cdn.tailwindcss.com; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.tailwindcss.com; "
    "font-src 'self' data: https://fonts.gstatic.com; "
    "img-src 'self' data: blob:; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "frame-ancestors 'none'; "
    "form-action 'self'"
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds hardening headers to every response."""
    async def dispatch(self, request: Request, call_next) -> Response:
        """Attach security headers, without overriding what a handler already set.

        setdefault rather than assignment, so an endpoint that needs a different
        policy can opt out. HSTS is omitted deliberately: it only means anything
        over HTTPS and is actively unhelpful during local development.
        """
        resp: Response = await call_next(request)
        resp.headers.setdefault("Content-Security-Policy", _CSP)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault("Permissions-Policy",
                                "camera=(), microphone=(), geolocation=()")
        # Strict-Transport-Security only useful behind HTTPS
        return resp


# Bearer-token auth (opt-in)

# Routes that bypass auth even when token is required (so the SPA can boot)
_PUBLIC_PREFIXES = ("/static/", "/api/info")
_PUBLIC_EXACT = {"/", "/index.html", "/favicon.ico"}


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """If RAG_AUTH_TOKEN is set in env, requires Authorization: Bearer <token>
    on every /api/* request except the public prefixes."""

    async def dispatch(self, request: Request, call_next) -> Response:
        """Enforce the bearer token on /api/* when one is configured.

        Static assets and /api/info stay public so the SPA can boot and show a
        login state rather than failing blank.
        """
        token = os.getenv("RAG_AUTH_TOKEN", "").strip()
        if not token:
            return await call_next(request)

        path = request.url.path
        if path in _PUBLIC_EXACT or any(path.startswith(p) for p in _PUBLIC_PREFIXES):
            return await call_next(request)
        if not path.startswith("/api/"):
            return await call_next(request)

        header = request.headers.get("authorization", "")
        if not header.lower().startswith("bearer "):
            return _json_error(401, "Missing bearer token.")
        provided = header.split(None, 1)[1].strip()
        # constant-time comparison
        import hmac
        if not hmac.compare_digest(provided, token):
            return _json_error(401, "Invalid bearer token.")
        return await call_next(request)


def _json_error(code: int, msg: str) -> Response:
    """JSON error response, matching FastAPI's {'detail': ...} shape."""
    from starlette.responses import JSONResponse
    return JSONResponse({"detail": msg}, status_code=code)


# Per-IP token-bucket rate limiter

class _Bucket:
    """One token bucket: `capacity` burst, refilling at `refill_per_sec`."""
    __slots__ = ("tokens", "last", "capacity", "refill_per_sec")

    def __init__(self, capacity: float, refill_per_sec: float) -> None:
        """Start full, so a first request is never delayed."""
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self.tokens = capacity
        self.last = time.monotonic()

    def take(self, n: float = 1.0) -> bool:
        """Consume n tokens, returning False if the bucket is short.

        Refill is computed lazily from elapsed time rather than by a timer, so an
        idle bucket costs nothing until it is next touched.
        """
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.refill_per_sec)
        self.last = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Cheap in-memory token-bucket per (client IP, route-class).
    Two classes: 'upload' (expensive) and 'default' (everything else)."""

    def __init__(self, app, *, default_rps: float = 5.0, default_burst: float = 30.0,
                 upload_rps: float = 0.2, upload_burst: float = 5.0) -> None:
        """Configure both buckets.

        Upload defaults are far tighter because one upload triggers parsing, vision
        captioning and a full embedding run.
        """
        super().__init__(app)
        self._lock = threading.Lock()
        self._buckets: dict[Tuple[str, str], _Bucket] = {}
        self._cfg = {
            "default": (default_burst, default_rps),
            "upload":  (upload_burst, upload_rps),
        }

    def _classify(self, request: Request) -> str:
        """Which bucket a request draws from: 'upload' or 'default'."""
        if request.method == "POST" and request.url.path == "/api/documents":
            return "upload"
        return "default"

    def _ip(self, request: Request) -> str:
        # Honor X-Forwarded-For when present (single-hop trust; tighten if behind LB)
        """Best-effort client identity.

        Trusts X-Forwarded-For when present, which is right behind a proxy that
        overwrites it and wrong when directly exposed — a client can then spoof the
        header to get a fresh bucket per request. Gate on a TRUST_PROXY flag before
        running without a reverse proxy.
        """
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            return xff.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        """Charge one token, or reject with 429. Only /api/* is limited.

        Buckets are created on demand and never evicted, so the dictionary grows
        with the number of distinct client IPs seen.
        """
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        cls = self._classify(request)
        key = (self._ip(request), cls)
        cap, rate = self._cfg[cls]
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(capacity=cap, refill_per_sec=rate)
                self._buckets[key] = bucket
            ok = bucket.take(1.0)
        if not ok:
            return _json_error(429, f"Rate limit exceeded ({cls}). Slow down.")
        return await call_next(request)


_DOC_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


def validate_doc_id(doc_id: str) -> str:
    """Reject anything that could escape the storage dir or be weird."""
    if not _DOC_ID_RE.match(doc_id):
        raise HTTPException(400, "Invalid document id.")
    # Defence in depth: no '..', no leading/trailing dots
    if ".." in doc_id or doc_id.startswith(".") or doc_id.endswith("."):
        raise HTTPException(400, "Invalid document id.")
    if PurePosixPath(doc_id).name != doc_id:
        raise HTTPException(400, "Invalid document id.")
    return doc_id


_FILENAME_BAD = re.compile(r"[^\w.\- ()]+")


def safe_filename(name: str | None) -> str:
    """Strip path components and weird chars from an uploaded filename."""
    name = (name or "upload.pdf").strip()
    name = name.replace("\\", "/").split("/")[-1]
    name = _FILENAME_BAD.sub("_", name)
    return name[:200] or "upload.pdf"


MAX_QUESTION_CHARS = 2000
MAX_HISTORY_TURNS = 20
MAX_HISTORY_CHAR_TOTAL = 32_000


def validate_chat_payload(question: str, history: list) -> None:
    """Bound question length and conversation size before any model call.

    Caps what an untrusted caller can push into the prompt budget. The history
    size check reads `.content` as an attribute, so it only measures objects —
    plain dicts silently contribute zero.
    """
    if not question or not question.strip():
        raise HTTPException(400, "Question is empty.")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(400, f"Question too long (>{MAX_QUESTION_CHARS} chars).")
    if len(history) > MAX_HISTORY_TURNS:
        raise HTTPException(400, f"History too long (>{MAX_HISTORY_TURNS} turns).")
    total = sum(len(getattr(h, "content", "") or "") for h in history)
    if total > MAX_HISTORY_CHAR_TOTAL:
        raise HTTPException(400, "Conversation history too large.")
