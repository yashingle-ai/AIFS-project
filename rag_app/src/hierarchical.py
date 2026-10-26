"""Hierarchical indexes and multi-document retrieval.

Three indexes per document, each a full dense + sparse pair:

  chunk     raw text, table and image chunks   precise facts, single cell values
  page      one summary per page               a metric plus its local context
  document  one summary per report             narrative, trends, "summarise this"

Why three tiers instead of one:
  A single chunk index answers "what was total income in H1 FY26" well and
  "summarise this report" badly — summarisation needs breadth that top-k over
  fragments cannot supply, and shrinking chunks to compensate wrecks the precise
  lookups. Rather than compromise chunk size, each question type gets an index
  built at its own granularity and route_query() in intelligence.py decides which
  tiers to search.

  The cost is real: three indexes per document, three builds, three loads.
  That trade only pays off because the tiers are cheap to build (page and
  document summaries are deterministic — see summarizer.py — so no LLM call is
  needed at index time).

Consistency trap:
  build() and load() both tolerate a missing page or document index, but
  index_exists() requires all three on disk. Callers must agree on which
  predicate they use — a document that produced no summaries will otherwise be
  re-indexed on every upload while remaining perfectly queryable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from . import rerank
from .chunker import Chunk
from .retriever import Hit, HybridRetriever


CHUNK_LEVEL = "chunk"
PAGE_LEVEL = "page"
DOCUMENT_LEVEL = "document"


@dataclass
class HierarchicalRetriever:
    """The three index tiers belonging to one document.

    page_index and document_index are optional: a document that produced no
    summaries still works, it just cannot answer summary-level questions well.
    """
    doc_id: str
    chunk_index: HybridRetriever
    page_index: Optional[HybridRetriever] = None
    document_index: Optional[HybridRetriever] = None

    @classmethod
    def build(
        cls,
        doc_id: str,
        chunks: List[Chunk],
        page_summary_chunks: List[Chunk],
        document_summary_chunks: List[Chunk],
    ) -> "HierarchicalRetriever":
        """Build all available tiers. Summary tiers are skipped when their input is empty."""
        chunk_index = HybridRetriever.build(chunks)
        page_index = HybridRetriever.build(page_summary_chunks) if page_summary_chunks else None
        document_index = HybridRetriever.build(document_summary_chunks) if document_summary_chunks else None
        return cls(
            doc_id=doc_id,
            chunk_index=chunk_index,
            page_index=page_index,
            document_index=document_index,
        )

    def save(self, dirpath: str | Path) -> None:
        """Write each tier; chunk tier at the root, summary tiers in subdirectories."""
        d = Path(dirpath)
        self.chunk_index.save(d)
        if self.page_index:
            self.page_index.save(d / "page_index")
        if self.document_index:
            self.document_index.save(d / "document_index")

    @classmethod
    def load(cls, doc_id: str, dirpath: str | Path) -> "HierarchicalRetriever":
        """Load whatever tiers exist on disk.

        Backfills doc_id onto chunks that predate multi-document support, so
        citations rendered from an older index still identify their source.
        """
        d = Path(dirpath)
        chunk_index = HybridRetriever.load(d)
        page_index = HybridRetriever.load(d / "page_index") if HybridRetriever.index_exists(d / "page_index") else None
        document_index = HybridRetriever.load(d / "document_index") if HybridRetriever.index_exists(d / "document_index") else None
        for index in (chunk_index, page_index, document_index):
            if not index:
                continue
            for chunk in index.chunks:
                if not chunk.doc_id:
                    chunk.doc_id = doc_id
        return cls(doc_id=doc_id, chunk_index=chunk_index, page_index=page_index, document_index=document_index)

    @staticmethod
    def index_exists(dirpath: str | Path) -> bool:
        """True only when all three tiers are on disk.

        Stricter than load(), which tolerates missing summary tiers. Callers that
        use this as a "has it been indexed" test will re-index documents that
        legitimately have no summaries — see the module docstring.
        """
        d = Path(dirpath)
        return (
            HybridRetriever.index_exists(d)
            and HybridRetriever.index_exists(d / "page_index")
            and HybridRetriever.index_exists(d / "document_index")
        )

    def search(self, query: str, level: str = CHUNK_LEVEL, k: int = 8, candidate_k: int = 30) -> List[Hit]:
        """Search one tier, falling back to the chunk tier if that tier is absent."""
        retriever = self._index_for_level(level)
        return retriever.search(query, k=k, candidate_k=max(candidate_k, k))

    def _index_for_level(self, level: str) -> HybridRetriever:
        """Resolve a level name to an index, defaulting to the chunk tier."""
        if level == PAGE_LEVEL and self.page_index:
            return self.page_index
        if level == DOCUMENT_LEVEL and self.document_index:
            return self.document_index
        return self.chunk_index


class MultiDocumentRetriever:
    """Fan-out across several documents' hierarchical indexes.

    Every query touches every document in scope, so cost grows linearly with
    corpus size. Fine for tens of documents; needs a routing layer beyond that.
    """
    def __init__(self, retrievers: Iterable[HierarchicalRetriever]):
        """Index the given retrievers by doc_id."""
        self.retrievers = {retriever.doc_id: retriever for retriever in retrievers}

    @property
    def doc_ids(self) -> List[str]:
        """Document ids currently in scope, sorted."""
        return sorted(self.retrievers)

    def filtered(self, doc_ids: Optional[Iterable[str]]) -> "MultiDocumentRetriever":
        """Narrow to a subset of documents. Empty/None means no filtering.

        Returns self when unfiltered, so callers must not mutate the result.
        """
        if not doc_ids:
            return self
        wanted = set(doc_ids)
        return MultiDocumentRetriever(r for doc_id, r in self.retrievers.items() if doc_id in wanted)

    def search(
        self,
        query: str,
        *,
        level: str = CHUNK_LEVEL,
        k: int = 8,
        per_doc_k: Optional[int] = None,
        candidate_k: int = 30,
    ) -> List[Hit]:
        """Search one tier across all in-scope documents and merge.

        Each document contributes its own top per_doc hits; duplicates are dropped
        by citation, then the cross-encoder picks the final top-k. Without the
        reranker the merge falls back to _global_score, which is much cruder.
        """
        if not self.retrievers:
            return []
        per_doc = per_doc_k or max(k, 6)
        candidates: List[Hit] = []
        for retriever in self.retrievers.values():
            candidates.extend(retriever.search(query, level=level, k=per_doc, candidate_k=candidate_k))
        candidates = _dedupe_hits(candidates)
        if rerank.is_enabled() and candidates:
            reranked = rerank.rerank(query, candidates, top_k=k)
            # rerank() never raises; it returns the input order when the backend
            # is unavailable. Confirm it actually scored something before
            # trusting the order, otherwise fall through to fusion ranking.
            if any(getattr(h, "rerank_score", 0.0) for h in reranked):
                return reranked
        # Explicit fallback: hybrid fusion ordering (see rerank.status()).
        return sorted(candidates, key=_global_score, reverse=True)[:k]

    def search_levels(
        self,
        query: str,
        *,
        levels: Iterable[str],
        k: int = 8,
        per_level_k: Optional[int] = None,
    ) -> List[Hit]:
        """Search several tiers and merge into one ranking.

        Used for summary and comparison intents, where a chunk-level answer alone
        would miss the surrounding narrative.
        """
        combined: List[Hit] = []
        for level in levels:
            combined.extend(self.search(query, level=level, k=per_level_k or k))
        return sorted(_dedupe_hits(combined), key=_global_score, reverse=True)[:k]


def _dedupe_hits(hits: List[Hit]) -> List[Hit]:
    """Drop repeat citations, keeping first occurrence (highest-ranked)."""
    out: List[Hit] = []
    seen: set[str] = set()
    for hit in hits:
        key = hit.chunk.citation
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
    return out


def _global_score(hit: Hit) -> float:
    """Comparable score for hits gathered from different rankings.

    Prefers the cross-encoder logit when reranking ran. Otherwise it blends
    cosine, RRF and a damped sparse term — these live on different scales, so
    the result is a heuristic ordering rather than a calibrated score. Dense
    similarity dominates in practice.
    """
    rerank_score = getattr(hit, "rerank_score", 0.0) or 0.0
    if rerank_score:
        return rerank_score
    sparse_component = min(max(hit.sparse_score, 0.0), 10.0) / 100.0
    return max(hit.dense_score, 0.0) + hit.score + sparse_component
