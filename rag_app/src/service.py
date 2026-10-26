"""Document registry + indexing service shared by the CLI and the web server.

Documents live under ./storage/<doc_id>/  containing:
  - source.pdf       the original upload
  - meta.json        { id, name, pages, chunks, status, error?, backend, model }
  - faiss.index      built by HybridRetriever.save()
  - bm25.pkl
  - chunks.json
  - summaries.json   deterministic page + document summaries
  - images/          extracted page images, served back by the API
  - page_index/, document_index/   the other two retrieval tiers

The filesystem IS the database. There is no metadata store: a document exists
because its directory exists, and meta.json is the record. That keeps the whole
system copyable and inspectable with nothing running, at the cost of having no
transactions — hence the status field below.

doc_id = sanitised filename + first 12 hex of the SHA1 of the file bytes.
Content addressing means re-uploading an identical PDF is a no-op rather than a
second index, while two genuinely different files with the same name still get
distinct directories.

Status is a state machine, not a boolean:
  pending -> indexing -> embedding -> ready
                      +-> error
Indexing takes minutes on a large PDF, so uploads return immediately and the UI
polls. Anything other than "ready" must be treated as not queryable — a
half-written index will load and return nonsense rather than fail loudly.

Concurrency caveats worth knowing before this is deployed for more than one user:
  - Each upload spawns its own daemon thread with no pool or queue, so N
    simultaneous uploads means N concurrent embedding runs.
  - embeddings._pace() throttles via a module global that is not lock-guarded,
    so those concurrent runs defeat the rate pacing they rely on.
  - _write_meta() is not atomic. A crash mid-write leaves meta.json truncated,
    _read_meta() swallows the parse error and the document silently disappears
    from list_documents(). Write to a temp file and os.replace() to fix.
  - The retriever caches below never evict; every document queried since start-up
    stays resident with its FAISS index.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from . import embeddings as _embeddings
from . import observability as _obs
from . import vision as _vision
from .chunker import CHUNKER_VERSION, chunk_pdf
from .enrichment import enrich_chunks
from .hierarchical import HierarchicalRetriever, MultiDocumentRetriever
from .ingest import extract_pdf
from .retriever import HybridRetriever
from .summarizer import (
    build_document_summary,
    build_page_summaries,
    save_summaries,
    summaries_to_chunks,
)


ROOT = Path(__file__).resolve().parent.parent
STORAGE = ROOT / "storage"
STORAGE.mkdir(exist_ok=True)


@dataclass
class DocMeta:
    """On-disk record for one document: identity, counts, status and provenance.

    backend/model/chunker are stored so a document indexed with different
    settings can be detected and rebuilt rather than silently mixed with newer
    ones in the same query.
    """
    id: str
    name: str
    pages: int = 0
    chunks: int = 0
    table_chunks: int = 0
    image_chunks: int = 0
    page_summaries: int = 0
    document_summaries: int = 0
    status: str = "pending"        # pending | indexing | embedding | ready | error
    stage: str = ""                # human-readable progress string for the UI
    error: Optional[str] = None
    backend: str = ""
    model: str = ""
    chunker: str = ""
    created_at: float = field(default_factory=time.time)


def _doc_dir(doc_id: str) -> Path:
    """Storage directory for a document."""
    return STORAGE / doc_id


def _meta_path(doc_id: str) -> Path:
    """Path to a document's meta.json."""
    return _doc_dir(doc_id) / "meta.json"


def _read_meta(doc_id: str) -> Optional[DocMeta]:
    """Load a document's metadata, or None if missing or unreadable.

    Swallows parse errors deliberately so one corrupt file cannot break the
    whole listing — but that also means a truncated write makes a document
    disappear silently. See the note on atomicity in the module docstring.
    """
    p = _meta_path(doc_id)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        allowed = set(DocMeta.__dataclass_fields__)
        return DocMeta(**{k: v for k, v in raw.items() if k in allowed})
    except Exception:
        return None


def _write_meta(meta: DocMeta) -> None:
    """Persist metadata.

    NOT atomic: a crash mid-write leaves a truncated file that _read_meta()
    will treat as a missing document. Write to a temp file and os.replace().
    """
    p = _meta_path(meta.id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, ensure_ascii=False, indent=2)


def list_documents() -> List[DocMeta]:
    """All documents on disk, newest first. Skips directories with no valid meta."""
    out: List[DocMeta] = []
    for d in sorted(STORAGE.iterdir(), key=lambda x: x.name):
        if not d.is_dir():
            continue
        meta = _read_meta(d.name)
        if meta:
            out.append(meta)
    out.sort(key=lambda m: m.created_at, reverse=True)
    return out


def get_document(doc_id: str) -> Optional[DocMeta]:
    """Metadata for one document, or None."""
    return _read_meta(doc_id)


def delete_document(doc_id: str) -> bool:
    """Remove a document and evict its cached retrievers.

    Cache eviction happens outside _cache_lock, so a concurrent get_retriever()
    can briefly resurrect a deleted entry.
    """
    d = _doc_dir(doc_id)
    if not d.exists():
        return False
    shutil.rmtree(d)
    _retriever_cache.pop(doc_id, None)
    _hierarchical_cache.pop(doc_id, None)
    return True


def pdf_path(doc_id: str) -> Optional[Path]:
    """Path to the stored source PDF, or None if absent."""
    p = _doc_dir(doc_id) / "source.pdf"
    return p if p.exists() else None


_IMG_NAME_RE = __import__("re").compile(r"^[A-Za-z0-9_]+\.(?:png|jpe?g|webp)$")


def image_path(doc_id: str, image_name: str) -> Optional[Path]:
    """Return the path to an image previously persisted under storage/{doc_id}/images/.
    Validates the filename to block any path traversal."""
    if not _IMG_NAME_RE.match(image_name):
        return None
    p = _doc_dir(doc_id) / "images" / image_name
    if not p.exists():
        return None
    # Defense in depth: ensure the resolved path is still inside the doc dir
    try:
        p.resolve().relative_to(_doc_dir(doc_id).resolve())
    except Exception:
        return None
    return p


# Indexing — runs in a background thread so the upload returns immediately.

_indexing_lock = threading.Lock()


def index_stale_reason(meta: DocMeta) -> Optional[str]:
    """Why a stored index cannot serve queries under the current config, or None.

    Vectors are only comparable within the embedding space that produced them.
    Swapping model or provider changes that space — and often the dimension too
    — so an index built earlier is not merely less accurate, it is unusable:
    FAISS raises a bare AssertionError on a width mismatch, deep inside the
    search call where the cause is invisible.

    This became real in Aug/Sep 2026 when NVIDIA retired nv-embedqa-e5-v5
    (1024-dim) and the replacement, nemotron-3-embed-1b, emits 2048-dim vectors.
    Checking metadata rather than probing the index keeps it free and offline.
    """
    current_backend = _embeddings.active_backend()
    current_model = _embeddings.active_model()
    if meta.backend and meta.backend != current_backend:
        return f"indexed with backend '{meta.backend}', now running '{current_backend}'"
    if meta.model and meta.model != current_model:
        return f"indexed with model '{meta.model}', now running '{current_model}'"
    return None


def _do_index(doc_id: str) -> None:
    """Run the full indexing pipeline for one document.

    extract -> caption images -> chunk -> enrich -> summarise -> embed -> save,
    writing meta after each stage so the polling UI can show progress.

    Runs on a background thread, so it must never raise: any failure is caught
    and recorded as status="error" with the message, otherwise the document
    would sit at "indexing" forever with no explanation.
    """
    meta = _read_meta(doc_id)
    if not meta:
        return
    try:
        meta.status = "indexing"
        meta.stage = "extracting"
        meta.backend = _embeddings.active_backend()
        meta.model = _embeddings.active_model()
        meta.chunker = CHUNKER_VERSION
        _write_meta(meta)

        pdf = _doc_dir(doc_id) / "source.pdf"
        pages = extract_pdf(pdf, with_images=_vision.is_enabled(), doc_id=doc_id)
        meta.pages = len(pages)
        n_imgs = sum(len(p.images) for p in pages)
        _write_meta(meta)

        # ---- Image captioning (vision LLM, optional) ----
        image_captions: dict = {}
        image_filenames: dict = {}
        if n_imgs and _vision.is_enabled():
            meta.stage = f"describing 0/{n_imgs} images"
            _write_meta(meta)

            # Persist image bytes to storage for the API endpoint to serve later
            img_dir = _doc_dir(doc_id) / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            flat: list[dict] = []
            keys: list[tuple[int, int]] = []
            for pg in pages:
                for img in pg.images:
                    fp = img_dir / img.filename
                    try:
                        fp.write_bytes(img.bytes)
                    except Exception:
                        continue
                    image_filenames[(img.page, img.index)] = img.filename
                    flat.append({"bytes": img.bytes, "mime": img.mime})
                    keys.append((img.page, img.index))

            def progress(done, total):
                # Update meta so the UI's polling endpoint shows live progress
                """Re-read, update and write the stage string so the UI can poll it."""
                m2 = _read_meta(doc_id)
                if m2:
                    m2.stage = f"describing {done}/{total} images"
                    _write_meta(m2)

            captions = _vision.describe_batch(flat, max_workers=4, on_progress=progress)
            for k, cap in zip(keys, captions):
                if cap:
                    image_captions[k] = cap

        # ---- Chunking ----
        meta.stage = "chunking"
        _write_meta(meta)
        chunks = chunk_pdf(pdf, pages, image_captions=image_captions, image_filenames=image_filenames)
        chunks = enrich_chunks(chunks, pages)
        meta.chunks = len(chunks)
        meta.table_chunks = sum(1 for c in chunks if c.kind == "table")
        meta.image_chunks = sum(1 for c in chunks if c.kind == "image")
        _write_meta(meta)

        # ---- Hierarchical summarization ----
        meta.stage = "summarizing pages and document"
        _write_meta(meta)
        page_summaries = build_page_summaries(pages, chunks)
        doc_summary = build_document_summary(doc_id, pages, page_summaries)
        save_summaries(_doc_dir(doc_id) / "summaries.json", page_summaries, doc_summary)
        page_summary_chunks, document_summary_chunks = summaries_to_chunks(page_summaries, doc_summary)
        meta.page_summaries = len(page_summary_chunks)
        meta.document_summaries = len(document_summary_chunks)
        _write_meta(meta)

        # ---- Embedding + hierarchical indexing ----
        meta.stage = (
            f"embedding {meta.chunks} chunks, "
            f"{meta.page_summaries} page summaries, "
            f"{meta.document_summaries} document summaries"
        )
        _write_meta(meta)
        retriever = HierarchicalRetriever.build(
            doc_id,
            chunks=chunks,
            page_summary_chunks=page_summary_chunks,
            document_summary_chunks=document_summary_chunks,
        )
        retriever.save(_doc_dir(doc_id))

        meta.status = "ready"
        meta.stage = ""
        meta.error = None
        _write_meta(meta)
    except Exception as e:
        meta = _read_meta(doc_id) or meta
        meta.status = "error"
        meta.error = str(e)[:500]
        meta.stage = ""
        _write_meta(meta)


def ingest_pdf(name: str, pdf_bytes: bytes) -> DocMeta:
    """Persist + kick off background indexing. Returns meta immediately."""
    h = hashlib.sha1(pdf_bytes).hexdigest()[:12]
    doc_id = f"{Path(name).stem.replace(' ', '_')[:40]}-{h}"
    d = _doc_dir(doc_id)

    existing = _read_meta(doc_id)
    if (
        existing
        and existing.status == "ready"
        and existing.chunker == CHUNKER_VERSION
        and not index_stale_reason(existing)
        and HierarchicalRetriever.index_exists(d)
    ):
        return existing  # exact same content already indexed

    d.mkdir(parents=True, exist_ok=True)
    (d / "source.pdf").write_bytes(pdf_bytes)

    meta = DocMeta(
        id=doc_id,
        name=name,
        status="pending",
        backend=_embeddings.active_backend(),
        model=_embeddings.active_model(),
        chunker=CHUNKER_VERSION,
    )
    _write_meta(meta)

    t = threading.Thread(target=_do_index, args=(doc_id,), daemon=True)
    t.start()
    return meta


# Retriever cache (one per doc_id, lazily loaded)

_retriever_cache: Dict[str, HybridRetriever] = {}
_hierarchical_cache: Dict[str, HierarchicalRetriever] = {}
_cache_lock = threading.Lock()


def _refuse_if_stale(doc_id: str) -> bool:
    """True when the stored index predates the active embedding config.

    Logs once per call rather than failing inside FAISS, so the cause appears in
    the logs as a sentence instead of an AssertionError with no message.
    """
    meta = _read_meta(doc_id)
    reason = index_stale_reason(meta) if meta else None
    if reason:
        _obs.log.warning(
            "index is stale and cannot be queried - re-upload the PDF to rebuild it",
            extra={"doc_id": doc_id, "reason": reason},
        )
        return True
    return False


def get_retriever(doc_id: str) -> Optional[HybridRetriever]:
    """Cached flat retriever for a document, or None if not indexed or stale."""
    with _cache_lock:
        if doc_id in _retriever_cache:
            return _retriever_cache[doc_id]
        d = _doc_dir(doc_id)
        if not HybridRetriever.index_exists(d) or _refuse_if_stale(doc_id):
            return None
        r = HybridRetriever.load(d)
        _retriever_cache[doc_id] = r
        return r


def get_hierarchical_retriever(doc_id: str) -> Optional[HierarchicalRetriever]:
    """Cached three-tier retriever for a document, or None if not indexed.

    Existence is tested on the chunk index alone, so a document with no summary
    tiers still loads. Note ingest_pdf() uses a stricter test — see the module
    docstring in hierarchical.py.
    """
    with _cache_lock:
        if doc_id in _hierarchical_cache:
            return _hierarchical_cache[doc_id]
        d = _doc_dir(doc_id)
        if not HybridRetriever.index_exists(d) or _refuse_if_stale(doc_id):
            return None
        r = HierarchicalRetriever.load(doc_id, d)
        _hierarchical_cache[doc_id] = r
        return r


def get_multi_document_retriever(doc_ids: Optional[Iterable[str]] = None) -> MultiDocumentRetriever:
    """Build a multi-document retriever over ready documents.

    Anything not in "ready" state is skipped: a partially built index would
    load and return misleading results rather than fail.
    """
    wanted = set(doc_ids or [])
    retrievers: List[HierarchicalRetriever] = []
    for meta in list_documents():
        if meta.status != "ready":
            continue
        if wanted and meta.id not in wanted:
            continue
        retriever = get_hierarchical_retriever(meta.id)
        if retriever:
            retrievers.append(retriever)
    return MultiDocumentRetriever(retrievers)


def reset_retriever_cache() -> None:
    """Drop all cached retrievers, forcing a reload from disk."""
    with _cache_lock:
        _retriever_cache.clear()
        _hierarchical_cache.clear()
