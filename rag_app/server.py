"""FastAPI backend for the document-grounded RAG.

Run:
    uvicorn server:app --reload --port 8000

Endpoints (all under /api):
  GET    /info                                    - active backends + models
  GET    /metrics                                 - Prometheus exposition
  GET    /documents                               - list indexed PDFs
  POST   /documents                               - upload a PDF (multipart 'file')
  DELETE /documents/{doc_id}                      - remove
  GET    /documents/{doc_id}                      - meta (poll for indexing status)
  GET    /documents/{doc_id}/pdf                  - serve the original PDF
  GET    /documents/{doc_id}/suggestions          - LLM-generated starter questions
  POST   /documents/{doc_id}/chat                 - ask a question (JSON, blocking)
  POST   /documents/{doc_id}/chat/stream          - ask a question (SSE streaming)
  POST   /feedback                                - thumbs up/down on an answer

Static SPA served at /  (rag_app/static/index.html).
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Load .env BEFORE importing modules that read env at import time
load_dotenv(Path(__file__).resolve().parent / ".env")

from src import embeddings as _embeddings
from src import llm as _llm
from src import observability as _obs
from src import rerank as _rerank
from src import service
from src.hierarchical import MultiDocumentRetriever
from src.intelligence import DocumentIntelligenceRAG
from src.security import (
    BearerAuthMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    cors_origins,
    safe_filename,
    validate_chat_payload,
    validate_doc_id,
)
from src.summarizer import load_summaries


_obs.configure_logging()

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
FEEDBACK_LOG = ROOT / "storage" / "feedback.jsonl"

app = FastAPI(title="Doc-Grounded RAG", version="1.1", docs_url=None, redoc_url=None)

# Order matters (request flows outer→inner; response inner→outer):
#   CORS → SecurityHeaders → BearerAuth → RateLimit → RequestLog → handler
app.add_middleware(_obs.RequestLogMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(BearerAuthMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class InfoResp(BaseModel):
    llm_backend: str
    llm_model: str
    embed_backend: str
    embed_model: str
    rerank_backend: str
    rerank_enabled: bool
    # "disabled" | "active" | "unavailable" — reflects what actually happens at
    # query time, not just what is configured. A retired endpoint reports
    # "unavailable" rather than silently looking like it is reranking.
    rerank_status: str
    rerank_unavailable_reason: str | None = None
    retrieval_mode: str


class DocResp(BaseModel):
    id: str
    name: str
    pages: int
    chunks: int
    table_chunks: int
    image_chunks: int = 0
    page_summaries: int = 0
    document_summaries: int = 0
    status: str
    stage: str = ""
    error: Optional[str] = None
    backend: str
    model: str
    chunker: str = ""
    created_at: float


class HistoryItem(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    question: str
    history: List[HistoryItem] = Field(default_factory=list)
    top_k: int = 8


class MultiChatReq(ChatReq):
    doc_ids: List[str] = Field(default_factory=list)


class HitResp(BaseModel):
    citation: str
    doc_id: str = ""
    page: int
    chunk_id: int
    kind: str
    content_type: str = ""
    section: str = ""
    text: str
    dense_score: float
    sparse_score: float
    score: float
    rerank_score: float = 0.0
    image_file: Optional[str] = None  # for kind="image" chunks; resolve via /api/documents/{id}/images/{file}


class UsageResp(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResp(BaseModel):
    question: str
    rewritten: str
    answer: str
    intent: str = ""
    hits: List[HitResp]
    usage: UsageResp = Field(default_factory=UsageResp)
    refused: bool = False


class FeedbackReq(BaseModel):
    doc_id: str
    question: str
    answer: str
    rating: str        # "up" | "down"
    comment: Optional[str] = None
    citations: List[str] = Field(default_factory=list)


class SuggestionsResp(BaseModel):
    questions: List[str]


class SummariesResp(BaseModel):
    document_summary: str
    page_summaries: List[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hits_to_resp(hits) -> List[HitResp]:
    return [
        HitResp(
            citation=h.chunk.citation,
            doc_id=getattr(h.chunk, "doc_id", ""),
            page=h.chunk.page,
            chunk_id=h.chunk.chunk_id,
            kind=h.chunk.kind,
            content_type=getattr(h.chunk, "content_type", "") or h.chunk.kind,
            section=getattr(h.chunk, "section", ""),
            text=h.chunk.text,
            dense_score=h.dense_score,
            sparse_score=h.sparse_score,
            score=h.score,
            rerank_score=getattr(h, "rerank_score", 0.0),
            image_file=getattr(h.chunk, "image_file", None),
        )
        for h in hits
    ]


def _record_chat_metrics(turn) -> None:
    backend = _llm.active_backend()
    model = _llm.active_model()
    _obs.chat_requests_total.labels(backend, model, str(bool(turn.refused)).lower()).inc()
    if turn.usage.get("prompt_tokens"):
        _obs.chat_tokens_total.labels(backend, "prompt").inc(turn.usage["prompt_tokens"])
    if turn.usage.get("completion_tokens"):
        _obs.chat_tokens_total.labels(backend, "completion").inc(turn.usage["completion_tokens"])
    if turn.hits:
        _obs.retrieval_top_dense_score.observe(max(h.dense_score for h in turn.hits))


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/api/info", response_model=InfoResp)
def api_info() -> InfoResp:
    return InfoResp(
        llm_backend=_llm.active_backend(),
        llm_model=_llm.active_model(),
        embed_backend=_embeddings.active_backend(),
        embed_model=_embeddings.active_model(),
        rerank_backend=_rerank.active_backend(),
        rerank_enabled=_rerank.is_enabled(),
        rerank_status=_rerank.status(),
        rerank_unavailable_reason=_rerank.unavailable_reason(),
        retrieval_mode=(
            "hybrid+rerank" if _rerank.is_enabled() else "hybrid (FAISS + BM25 + RRF)"
        ),
    )


@app.get("/api/metrics")
def api_metrics():
    # Refresh the docs gauge before scraping
    try:
        ready = sum(1 for d in service.list_documents() if d.status == "ready")
        _obs.documents_indexed.set(ready)
    except Exception:
        pass
    return _obs.metrics_response()


@app.get("/api/documents", response_model=List[DocResp])
def api_list_documents() -> List[DocResp]:
    return [DocResp(**vars(m)) for m in service.list_documents()]


@app.post("/api/documents", response_model=DocResp)
async def api_upload(file: UploadFile = File(...)) -> DocResp:
    fname = safe_filename(file.filename)
    if not fname.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    MAX = 50 * 1024 * 1024
    data = await file.read(MAX + 1)
    if not data:
        raise HTTPException(400, "Empty file.")
    if len(data) > MAX:
        raise HTTPException(400, "File too large (>50MB).")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(400, "Not a valid PDF file.")
    meta = service.ingest_pdf(fname, data)
    _obs.log.info("doc_upload",
                  extra={"doc_id": meta.id, "doc_name": meta.name, "size_bytes": len(data)})
    return DocResp(**vars(meta))


@app.get("/api/documents/{doc_id}", response_model=DocResp)
def api_get_doc(doc_id: str) -> DocResp:
    doc_id = validate_doc_id(doc_id)
    m = service.get_document(doc_id)
    if not m:
        raise HTTPException(404, "Not found")
    return DocResp(**vars(m))


@app.delete("/api/documents/{doc_id}")
def api_delete(doc_id: str) -> JSONResponse:
    doc_id = validate_doc_id(doc_id)
    ok = service.delete_document(doc_id)
    if not ok:
        raise HTTPException(404, "Not found")
    _obs.log.info("doc_delete", extra={"doc_id": doc_id})
    return JSONResponse({"ok": True})


@app.get("/api/documents/{doc_id}/pdf")
def api_pdf(doc_id: str):
    doc_id = validate_doc_id(doc_id)
    p = service.pdf_path(doc_id)
    if not p:
        raise HTTPException(404, "Not found")
    return FileResponse(p, media_type="application/pdf",
                        headers={"Content-Disposition": "inline"})


@app.get("/api/documents/{doc_id}/images/{image_name}")
def api_image(doc_id: str, image_name: str):
    doc_id = validate_doc_id(doc_id)
    p = service.image_path(doc_id, image_name)
    if not p:
        raise HTTPException(404, "Image not found")
    ext = p.suffix.lower().lstrip(".")
    mime = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png","webp":"image/webp"}.get(ext, "application/octet-stream")
    return FileResponse(p, media_type=mime,
                        headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/documents/{doc_id}/suggestions", response_model=SuggestionsResp)
def api_suggestions(doc_id: str) -> SuggestionsResp:
    """Generate 4 starter questions from a sample of the document's chunks.
    Cached on disk so we only hit the LLM once per doc."""
    doc_id = validate_doc_id(doc_id)
    meta = service.get_document(doc_id)
    if not meta:
        raise HTTPException(404, "Document not found")
    if meta.status != "ready":
        raise HTTPException(409, f"Document not ready (status={meta.status}).")

    cache = ROOT / "storage" / doc_id / "suggestions.json"
    if cache.exists():
        try:
            return SuggestionsResp(questions=json.loads(cache.read_text())[:4])
        except Exception:
            pass

    retriever = service.get_retriever(doc_id)
    if not retriever:
        raise HTTPException(500, "Retriever could not be loaded.")

    # Sample first 8 chunks evenly across the doc
    chunks = retriever.chunks
    if not chunks:
        return SuggestionsResp(questions=[])
    step = max(1, len(chunks) // 8)
    sample = [c.text[:500] for c in chunks[::step][:8]]
    sample_block = "\n\n---\n\n".join(sample)

    prompt = (
        "Below are excerpts from a document. Propose 4 short, distinct questions "
        "a curious reader would ask, each grounded in the excerpts. Return ONLY "
        "the questions, one per line, no numbering, no quotes.\n\n"
        f"EXCERPTS:\n{sample_block}\n\nQUESTIONS:"
    )
    try:
        out = _llm._generate_text(  # noqa: SLF001 — internal use is fine here
            "You generate concise starter questions from document excerpts.",
            prompt, temperature=0.3,
        )
    except Exception as e:
        _obs.log.warning("suggestions_failed", extra={"doc_id": doc_id, "err": str(e)})
        return SuggestionsResp(questions=[])

    qs = [q.strip(" -•*\"'") for q in out.splitlines()]
    qs = [q for q in qs if q and len(q) > 8 and q.endswith("?")][:4]
    if qs:
        try: cache.write_text(json.dumps(qs))
        except Exception: pass
    return SuggestionsResp(questions=qs)


@app.get("/api/documents/{doc_id}/summaries", response_model=SummariesResp)
def api_summaries(doc_id: str) -> SummariesResp:
    doc_id = validate_doc_id(doc_id)
    meta = service.get_document(doc_id)
    if not meta:
        raise HTTPException(404, "Document not found")
    if meta.status != "ready":
        raise HTTPException(409, f"Document not ready (status={meta.status}).")
    path = ROOT / "storage" / doc_id / "summaries.json"
    if not path.exists():
        raise HTTPException(404, "Summaries not found. Rebuild the document index.")
    page_summaries, doc_summary = load_summaries(path)
    return SummariesResp(
        document_summary=doc_summary.to_markdown(),
        page_summaries=[summary.to_markdown() for summary in page_summaries],
    )


@app.post("/api/chat", response_model=ChatResp)
def api_multi_chat(req: MultiChatReq) -> ChatResp:
    """Ask across all ready documents, or a supplied doc_ids subset."""
    validate_chat_payload(req.question, req.history)
    doc_ids = [validate_doc_id(doc_id) for doc_id in req.doc_ids]
    for doc_id in doc_ids:
        meta = service.get_document(doc_id)
        if not meta:
            raise HTTPException(404, f"Document not found: {doc_id}")
        if meta.status != "ready":
            raise HTTPException(409, f"Document not ready: {doc_id} (status={meta.status}).")

    multi = service.get_multi_document_retriever(doc_ids or None)
    if not multi.doc_ids:
        raise HTTPException(404, "No ready documents found.")

    rag = DocumentIntelligenceRAG(retriever=multi, top_k=max(1, min(req.top_k, 16)))
    rag.history = [{"role": h.role, "content": h.content} for h in req.history]
    try:
        turn = rag.ask(req.question, doc_ids=doc_ids or None)
    except Exception as e:
        _obs.log.exception("multi_chat_error")
        raise HTTPException(502, f"Upstream LLM error: {e}")

    _record_chat_metrics(turn)
    _obs.log.info("multi_chat", extra={
        "doc_ids": ",".join(multi.doc_ids), "intent": turn.intent,
        "refused": turn.refused, "hits": len(turn.hits),
        "tokens": turn.usage.get("total_tokens", 0),
    })

    return ChatResp(
        question=turn.question,
        rewritten=turn.rewritten,
        answer=turn.answer,
        intent=turn.intent,
        hits=_hits_to_resp(turn.hits),
        usage=UsageResp(**turn.usage),
        refused=turn.refused,
    )


@app.post("/api/documents/{doc_id}/chat", response_model=ChatResp)
def api_chat(doc_id: str, req: ChatReq) -> ChatResp:
    doc_id = validate_doc_id(doc_id)
    validate_chat_payload(req.question, req.history)
    meta = service.get_document(doc_id)
    if not meta:
        raise HTTPException(404, "Document not found")
    if meta.status != "ready":
        raise HTTPException(409, f"Document not ready (status={meta.status}).")

    retriever = service.get_hierarchical_retriever(doc_id)
    if not retriever:
        raise HTTPException(500, "Retriever could not be loaded.")

    rag = DocumentIntelligenceRAG(
        retriever=MultiDocumentRetriever([retriever]),
        top_k=max(1, min(req.top_k, 16)),
    )
    rag.history = [{"role": h.role, "content": h.content} for h in req.history]
    try:
        turn = rag.ask(req.question, doc_ids=[doc_id])
    except Exception as e:
        _obs.log.exception("chat_error", extra={"doc_id": doc_id})
        raise HTTPException(502, f"Upstream LLM error: {e}")

    _record_chat_metrics(turn)
    _obs.log.info("chat", extra={
        "doc_id": doc_id, "refused": turn.refused,
        "hits": len(turn.hits), "tokens": turn.usage.get("total_tokens", 0),
    })

    return ChatResp(
        question=turn.question,
        rewritten=turn.rewritten,
        answer=turn.answer,
        intent=turn.intent,
        hits=_hits_to_resp(turn.hits),
        usage=UsageResp(**turn.usage),
        refused=turn.refused,
    )


@app.post("/api/documents/{doc_id}/chat/stream")
async def api_chat_stream(doc_id: str, req: ChatReq):
    """SSE stream. Frame schema:
       event: meta   data: {hits:[...], rewritten, refused}
       event: token  data: {t: "<text fragment>"}
       event: done   data: {usage:{...}}
       event: error  data: {detail: "..."}
    All `data:` payloads are JSON. We stream the answer in word-sized chunks (the
    upstream LLM call itself isn't streamed because we do retrieval+refusal logic
    first; but the user still sees a 'typing' effect that paints the answer).
    """
    doc_id = validate_doc_id(doc_id)
    validate_chat_payload(req.question, req.history)
    meta = service.get_document(doc_id)
    if not meta:
        raise HTTPException(404, "Document not found")
    if meta.status != "ready":
        raise HTTPException(409, f"Document not ready (status={meta.status}).")
    retriever = service.get_hierarchical_retriever(doc_id)
    if not retriever:
        raise HTTPException(500, "Retriever could not be loaded.")

    rag = DocumentIntelligenceRAG(
        retriever=MultiDocumentRetriever([retriever]),
        top_k=max(1, min(req.top_k, 16)),
    )
    rag.history = [{"role": h.role, "content": h.content} for h in req.history]

    def sse(event: str, payload: dict) -> bytes:
        return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode("utf-8")

    async def gen():
        loop = asyncio.get_running_loop()
        try:
            # Run the (blocking) ask in a worker thread so we can yield meanwhile.
            turn = await loop.run_in_executor(None, rag.ask, req.question, [doc_id])

            # 1) meta frame: send hits + rewritten so the UI can show the debug
            #    panel + confidence pill before the answer paints.
            yield sse("meta", {
                "rewritten": turn.rewritten,
                "intent": turn.intent,
                "refused": turn.refused,
                "hits": [h.dict() for h in _hits_to_resp(turn.hits)],
            })

            # 2) stream the answer in small chunks for that "typing" effect
            text = turn.answer or ""
            i = 0
            CHUNK = 24  # chars
            while i < len(text):
                yield sse("token", {"t": text[i:i + CHUNK]})
                i += CHUNK
                await asyncio.sleep(0.012)

            # 3) done frame with usage
            yield sse("done", {"usage": turn.usage})
            _record_chat_metrics(turn)
            _obs.log.info("chat_stream", extra={
                "doc_id": doc_id, "refused": turn.refused,
                "hits": len(turn.hits), "tokens": turn.usage.get("total_tokens", 0),
            })
        except Exception as e:
            _obs.log.exception("chat_stream_error", extra={"doc_id": doc_id})
            yield sse("error", {"detail": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",  # disable nginx buffering if proxied
    })


@app.post("/api/feedback")
async def api_feedback(req: FeedbackReq):
    if req.rating not in ("up", "down"):
        raise HTTPException(400, "rating must be 'up' or 'down'.")
    if len(req.question) > 4000 or len(req.answer) > 8000 or len(req.comment or "") > 2000:
        raise HTTPException(400, "feedback payload too large.")
    rec = {
        "ts": time.time(),
        "doc_id": req.doc_id,
        "rating": req.rating,
        "question": req.question[:2000],
        "answer": req.answer[:4000],
        "comment": (req.comment or "")[:1000],
        "citations": req.citations[:20],
        "request_id": _obs.current_request_id(),
    }
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _obs.feedback_total.labels(req.rating).inc()
    _obs.log.info("feedback", extra={"rating": req.rating, "doc_id": req.doc_id})
    return {"ok": True}


# ---------------------------------------------------------------------------
# Static SPA — must be last so /api/* routes win.
# ---------------------------------------------------------------------------

if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
