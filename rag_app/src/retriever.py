"""Hybrid retriever: dense (FAISS, cosine via inner-product) + sparse (BM25),
fused with Reciprocal Rank Fusion. Persists to disk so re-running is instant.

Why hybrid and not dense alone:
  Financial questions carry both halves of the problem in one sentence. "Q2 FY26
  revenue" needs an exact lexical hit on a period string that dense embeddings
  smear together with every other quarter, while "how did the airport segment
  perform" needs semantic matching that BM25 cannot do at all. Neither index
  answers both, so both run and the rankings are merged.

Why IndexFlatIP rather than an ANN index:
  Flat means exhaustive — every query is compared against every vector, so recall
  is 100% by construction. HNSW and IVF exist to trade recall for speed on
  corpora far larger than one financial report. At a few hundred to a few
  thousand chunks there is no speed problem to buy, so approximating would cost
  recall for nothing. In this domain a missed chunk is a wrong or refused answer.
  Swap to IndexHNSWFlat around ~1M vectors, not before.

  IP (inner product) is cosine similarity here only because embeddings.py
  L2-normalizes every vector first.

Why RRF instead of blending the scores:
  Cosine similarity and BM25 scores live on different, unnormalised scales and
  cannot be added meaningfully. RRF throws the magnitudes away and fuses on RANK
  alone, so neither index needs calibrating against the other. The k=60 constant
  is the value from the original TREC work; it flattens the contribution curve so
  a single index cannot dominate purely by being confident.

  The raw dense/sparse scores are still carried on each Hit, because the refusal
  floor upstream needs absolute confidence, not relative rank.

Persistence:
  Three files per index — vectors, BM25 state, and the chunk texts they point at.
  All three must be written together or a reload will mismatch ids to text.

  NOTE: bm25.pkl is loaded with pickle, which executes arbitrary code on load.
  That is fine while storage/ is local and trusted; it would not be acceptable
  in a multi-tenant deployment where another user can write to that directory.
"""
from __future__ import annotations

import json
import pickle
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from . import observability as _obs
from .chunker import Chunk
from .embeddings import embed_documents, embed_query


_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_/%.]*")


def _tokenize(text: str) -> List[str]:
    """Lowercase tokens for BM25.

    The character class keeps -_/%. inside a token so "Q2/FY26", "nv-embedqa"
    and "12.5%" survive as single terms. Splitting those apart would destroy
    exactly the lexical anchors sparse retrieval exists to catch.
    """
    return [t.lower() for t in _TOKEN.findall(text)]


@dataclass
class Hit:
    """One retrieved chunk plus every score that produced it.

    All four scores are kept rather than just the fused one because they answer
    different questions: `score` orders results, `dense_score` and
    `sparse_score` feed the refusal floor (which needs absolute confidence, not
    rank), and `rerank_score` is set only when the cross-encoder ran.
    """
    chunk: Chunk
    score: float                # fused score (higher = better)
    dense_score: float          # cosine similarity in [-1, 1]
    sparse_score: float         # raw bm25 score
    rerank_score: float = 0.0   # cross-encoder logit (set by reranker if enabled)


class HybridRetriever:
    """Dense + sparse index pair over one set of chunks.

    The three pieces (FAISS vectors, BM25 state, chunk texts) are positionally
    aligned: row i of the index, document i of BM25 and chunks[i] are the same
    chunk. Nothing enforces that invariant, so they must always be built and
    persisted together.
    """
    def __init__(self, chunks: List[Chunk], index: faiss.Index, bm25: BM25Okapi):
        """Wrap an already-built index triple. Use build() or load() instead."""
        self.chunks = chunks
        self.index = index
        self.bm25 = bm25

    # ---- build & persist ----

    @classmethod
    def build(cls, chunks: List[Chunk]) -> "HybridRetriever":
        """Embed the chunks and construct both indexes.

        This is the expensive call — it embeds every chunk, so cost is linear in
        corpus size and billed per token. Callers persist the result with save()
        and reload it rather than rebuilding.
        """
        if not chunks:
            raise ValueError("No chunks to index.")
        texts = [c.text for c in chunks]
        vecs = embed_documents(texts)
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        bm25 = BM25Okapi([_tokenize(t) for t in texts])
        return cls(chunks=chunks, index=index, bm25=bm25)

    def save(self, dirpath: str | Path) -> None:
        """Persist all three components into `dirpath`.

        Partial writes leave the index unusable: BM25 and FAISS positions are only
        meaningful against the chunks.json written in the same call.
        """
        d = Path(dirpath)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "faiss.index"))
        with open(d / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        with open(d / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, ensure_ascii=False)

    @classmethod
    def load(cls, dirpath: str | Path) -> "HybridRetriever":
        """Restore a retriever previously written by save().

        Note this unpickles bm25.pkl, which executes arbitrary code — safe only
        while storage/ is a trusted local directory.
        """
        d = Path(dirpath)
        index = faiss.read_index(str(d / "faiss.index"))
        with open(d / "bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)
        with open(d / "chunks.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        chunks = [Chunk(**r) for r in raw]
        return cls(chunks=chunks, index=index, bm25=bm25)

    @staticmethod
    def index_exists(dirpath: str | Path) -> bool:
        """True when all three files are present, i.e. load() will succeed."""
        d = Path(dirpath)
        return all((d / n).exists() for n in ("faiss.index", "bm25.pkl", "chunks.json"))

    # ---- search ----

    def _dense(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Top-k by cosine similarity from FAISS.

        Returns (chunk_index, score) pairs. FAISS pads with -1 when the corpus has
        fewer than k vectors, so those entries are dropped.
        """
        qv = embed_query(query)
        scores, ids = self.index.search(qv, min(k, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def _sparse(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Top-k by BM25 lexical overlap.

        Returns (chunk_index, score) pairs, dropping zero scores — BM25 assigns
        0 to anything sharing no term with the query, and those carry no signal.
        """
        toks = _tokenize(query)
        if not toks:
            return []
        # get_scores() walks the entire corpus in Python and returns one score per
        # chunk, so this is O(N) per query with no inverted-index shortcut. It is
        # the first thing that will fall over as the corpus grows — well before
        # FAISS, which does its scan in vectorised C++. Move sparse retrieval to a
        # real inverted index (OpenSearch, Tantivy) before scaling past ~100k chunks.
        scores = self.bm25.get_scores(toks)
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def search(self, query: str, k: int = 6, candidate_k: int = 20, rrf_k: int = 60) -> List[Hit]:
        """Return top-k chunks via RRF over dense + sparse rankings."""
        started = time.perf_counter()
        dense = self._dense(query, candidate_k)
        sparse = self._sparse(query, candidate_k)

        d_scores = {i: s for i, s in dense}
        s_scores = {i: s for i, s in sparse}

        rrf: dict[int, float] = {}
        for rank, (i, _) in enumerate(dense):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, (i, _) in enumerate(sparse):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)

        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]
        _obs.retrieval_duration_seconds.observe(time.perf_counter() - started)
        return [
            Hit(
                chunk=self.chunks[i],
                score=score,
                dense_score=d_scores.get(i, 0.0),
                sparse_score=s_scores.get(i, 0.0),
            )
            for i, score in ranked
        ]
