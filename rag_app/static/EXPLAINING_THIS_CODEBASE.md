# Explaining QuantX — a study and interview guide

How to read this codebase, in the order that makes it make sense, plus the
concepts you need behind each file to defend it under questioning.

Work top to bottom. Each stage names the files, what to be able to say about
them, and what to go learn if you cannot say it yet.

---

## 0. The 60-second version

Learn this first. Everything else is elaboration.

> QuantX answers questions about large financial PDFs — earnings decks, filings,
> annual reports — and every claim in the answer carries a citation back to the
> page it came from. It handles text, tables and charts, and when the document
> does not support an answer it says so instead of guessing.
>
> The hard parts were: tables lose meaning when you chunk them naively, charts
> are invisible to text-only retrieval, and financial questions need both exact
> keyword matching and semantic matching at once.

If someone asks only one follow-up it will be **"why is this more than a vector
database with a chat box?"**. The answer is the three-tier hierarchy, the hybrid
retrieval, the vision pass, and the refusal floor — in that order of impressiveness.

---

## 1. The shape of the system

Two pipelines that meet at the indexes.

```
INGESTION (once per document, minutes)
  PDF -> ingest.py -> chunker.py -> vision.py -> enrichment.py
      -> summarizer.py -> embeddings.py -> retriever.py (saved to disk)

QUERY (per question, seconds)
  question -> intelligence.py (route) -> chat.py (fiscal expansion)
           -> retriever.py (FAISS + BM25 -> RRF) -> rerank.py
           -> qwen_summarizer.py (optional) -> llm.py -> citations -> answer
```

**Say this:** indexing is expensive and happens once; querying is cheap and
happens constantly. Every design decision follows from pushing work into the
first pipeline. Chart captioning is the clearest example — a vision model reads
each chart at index time, so queries pay nothing for multimodal support.

---

## 2. Read the files in this order

Not alphabetical, not by size. This order builds understanding.

| # | File | What it does | Read it for |
|---|---|---|---|
| 1 | `ingest.py` | PDF -> pages, text, tables, images | how a PDF is actually structured |
| 2 | `chunker.py` | pages -> chunks | why chunking is the whole ballgame |
| 3 | `vision.py` | images -> text captions | the multimodal trick |
| 4 | `enrichment.py` | chunks -> chunks + extracted facts | making tables retrievable |
| 5 | `summarizer.py` | pages -> page/document summaries | how tier 2 and 3 get built |
| 6 | `embeddings.py` | text -> vectors | why normalisation matters |
| 7 | `retriever.py` | **the core** — FAISS + BM25 + RRF | hybrid search |
| 8 | `rerank.py` | cross-encoder precision pass | bi-encoder vs cross-encoder |
| 9 | `hierarchical.py` | three tiers, many documents | why one index is not enough |
| 10 | `chat.py` | single-doc conversation, fiscal expansion | the domain differentiator |
| 11 | `intelligence.py` | routing, refusal, citation repair | the decision layer |
| 12 | `llm.py` | prompts and generation | grounding via prompt |
| 13 | `service.py` | document lifecycle, storage | how it holds together |
| 14 | `security.py` / `observability.py` | hardening, metrics | production concerns |

**If you only have time for three:** `retriever.py`, `intelligence.py`,
`chunker.py`. Those three carry almost every interesting question.

---

## 3. Concepts, per stage

### Stage 1 — Ingestion (`ingest.py`, `chunker.py`, `table_reconstruct.py`)

**Learn:**
- Why a PDF has no reading order. It is a set of positioned glyphs, not a
  document tree. Multi-column layouts read left-to-right across columns unless
  something reconstructs the order.
- Chunking strategies: fixed-size, recursive, semantic, layout-aware. Know the
  trade-offs and which this project uses (layout-aware via OpenDataLoader, with
  a PyMuPDF fallback).
- Chunk overlap — what it is and why it exists.
- Why tables break naive chunking: a split mid-table orphans rows from headers,
  so a number survives but the label describing it does not.

**Be able to say:** *"We treat tables as atomic and replay the header row into
every split — `_split_table_markdown` in chunker.py. Without that, a chunk
containing '469.80' has no idea whether that is finance cost or revenue."*

**Also know:** `table_reconstruct.py` rebuilds tables from raw glyph coordinates
when both parsers fail — grouping spans into rows by y-position, clustering
x-positions into columns. This is the most technically impressive file in the
repo. Mention it.

### Stage 2 — Multimodal (`vision.py`)

**Learn:**
- What a vision-language model is and how images are passed (base64 data URLs).
- Why captioning at index time beats captioning at query time: cost and latency
  are paid once, and the caption becomes an ordinary searchable chunk.
- Cost control — `is_meaningful_image()` filters logos and dividers by size and
  aspect ratio so you do not pay tokens for page furniture.

**Be able to say:** *"A bar chart is invisible to a text-only pipeline. We caption
it into a markdown table at index time, so it is retrievable by both FAISS and
BM25 like any other chunk, at zero added query latency."*

### Stage 3 — Embeddings (`embeddings.py`)

**Learn:**
- What an embedding is; what dimensionality means (1024 here).
- Cosine similarity, and why L2-normalising makes inner product equal cosine.
- **Asymmetric embedding** — passages and queries encoded differently. The E5
  family requires it. This is a genuine "they know their stuff" detail.
- Batching and rate limiting against a hosted API.

**Be able to say:** *"We L2-normalise on the way out, which is what lets FAISS
IndexFlatIP behave as a cosine index and keeps dense_score in a fixed range the
refusal threshold can compare against."*

### Stage 4 — Retrieval (`retriever.py`) ← **the most important file**

**Learn, in order:**

1. **Vector search** — dense retrieval, ANN vs exact.
2. **FAISS index types** — `IndexFlatIP` (exact, brute force) vs `IndexHNSWFlat`
   / `IVFPQ` (approximate). Know that flat = 100% recall, and that ANN exists to
   trade recall for speed on much larger corpora.
3. **BM25** — term frequency, inverse document frequency, length normalisation.
   Enough to explain why it catches "Q2 FY26" when embeddings smear all quarters
   together.
4. **Hybrid search** — why neither index alone is sufficient.
5. **Reciprocal Rank Fusion** — `score = Σ 1/(k + rank)`, k=60. Crucially: RRF
   fuses on *rank*, not score, so two incomparable scoring scales never have to
   be calibrated against each other.

**The question you will get:** *"Why FAISS and not Pinecone/Chroma/Qdrant?"*

> "FAISS is a similarity-search library, not a database — no filtering, no CRUD,
> no persistence layer. We supplied those: chunks.json is the document store,
> one folder per document is the namespace. At a few hundred to a few thousand
> chunks per document, exact search is fast and gives perfect recall, so an
> approximate index would trade away correctness for a speed win we did not
> need. I would move to Qdrant or Weaviate when we need metadata filtering,
> deletes without full rebuilds, or multi-process serving."

**Follow-up you should volunteer:** the pure-Python BM25 (`get_scores` is O(N)
over the whole corpus) falls over before FAISS does. Naming your own bottleneck
lands very well.

### Stage 5 — Reranking (`rerank.py`)

**Learn this properly, it is a favourite interview topic:**

- **Bi-encoder** (what embeddings do): question and document are encoded
  *separately* into vectors, compared by cosine. Fast, because documents are
  encoded once at index time — but the model never sees the pair together.
- **Cross-encoder** (what the reranker does): question and document are fed
  through the model *together*, so attention runs across both. Much more
  accurate, and far too slow to run over the whole corpus.
- **Therefore the two-stage pattern:** retrieve ~30 candidates cheaply with the
  bi-encoder, then re-score just those with the cross-encoder and keep the top
  few. Recall from stage one, precision from stage two.

**Be able to say:** *"Retrieval optimises recall, reranking optimises precision.
We pull 30 candidates via hybrid search, then a cross-encoder re-scores them
because it can attend over the question and passage jointly — something a
bi-encoder structurally cannot do. We keep the top-k of that."*

**Know the trade-off:** it costs an extra API round trip per query. Our reranker
fails open — on any error it returns the original ordering rather than breaking
the request.

### Stage 6 — Hierarchy (`hierarchical.py`, `summarizer.py`)

**Learn:** why one index cannot serve every question type.

| Tier | Content | Answers |
|---|---|---|
| chunk | raw text, tables, image captions | "what was total income in H1 FY26?" |
| page | one summary per page | a metric plus its surrounding context |
| document | one summary per report | "summarise this report" |

**Be able to say:** *"Top-k over fragments cannot summarise a 300-page report —
you get fragments back. Rather than compromise chunk size, each question type
gets an index at its own granularity, and the router picks the tier."*

**Cost note worth volunteering:** the summaries are *deterministic* — extractive
sentence ranking in `summarizer.py`, no LLM call — so three tiers cost almost
nothing extra at index time and are perfectly reproducible.

### Stage 7 — Query understanding (`chat.py`, `intelligence.py`)

**Learn:**
- **Query rewriting / decomposition** — resolving "what about the next quarter?"
  into a standalone query, and splitting multi-part questions. Retrieval has no
  conversational memory; an unresolved pronoun retrieves nothing.
- **Intent routing** — different question types need different retrieval
  strategies and different top-k.
- **Domain query expansion** — the fiscal calendar work. `Q2 FY26` →
  `quarter ended 30-09-2025`. This is the single most concrete differentiator in
  the project, because it is a real domain problem no general-purpose RAG solves.

**Be able to say:** *"Users ask for Q2 FY26; the filing says 'quarter ended
30-09-2025'. Same meaning, zero lexical overlap, and the embedding does not
reliably bridge it either. We expand the retrieval query with the date-ended
phrasing while leaving the user's question untouched — we widen recall without
changing what the model is asked."*

### Stage 8 — Generation and grounding (`llm.py`, `intelligence.py`)

**Learn:**
- Prompt engineering for grounding: "use only the context", explicit refusal
  instruction, mandatory citations.
- Why temperature 0 for extraction tasks.
- **Citation validation and repair** — models emit plausible but wrong citation
  ids; `_repair_citations` maps them back onto chunks actually retrieved and
  drops the ones that cannot be resolved.
- **The refusal floor** — if the best dense and best sparse scores are both
  below threshold, refuse *before* calling the model.

**Be able to say:** *"Grounding is three layers, not a prompt. The prompt forbids
outside knowledge; a score floor refuses before we even call the model, which
removes the opportunity to answer from parametric memory; and citations are
validated against retrieved chunks afterwards. Any one alone is not a guarantee."*

### Stage 9 — Production concerns (`service.py`, `security.py`, `observability.py`)

**Learn:**
- Background jobs and why upload returns immediately with a polled status.
- Content-addressed storage — SHA1 of file bytes for deduplication.
- **SSE (server-sent events)** vs WebSockets for token streaming.
- Async vs sync in FastAPI: `def` handlers run in a threadpool, `async def` run
  on the event loop, so blocking work inside `async def` needs `run_in_executor`.
- Token-bucket rate limiting.
- Prometheus metric types (counter, gauge, histogram) and **label cardinality** —
  why a document id must never become a label.
- Structured logging and request-id correlation.

**Be able to say:** *"Indexing takes minutes, so uploads return immediately and
the client polls a status field. Answers stream over SSE. Blocking calls run in
FastAPI's threadpool so the event loop is never held."*

---

## 4. Questions you will actually be asked

**"Walk me through what happens when a user asks a question."**
Follow the query pipeline in §1. Name the file at each hop. Do not skip the
refusal check — it is the most distinctive step.

**"How do you prevent hallucination?"**
Three layers: prompt constraints, the pre-generation score floor, post-hoc
citation validation. Say that the score floor matters most, because it removes
the *opportunity* rather than instructing against it.

**"Why hybrid search?"**
Semantic misses exact tokens; lexical misses paraphrase. Financial questions
need both in one sentence. Give the "Q2 FY26 revenue" example.

**"What is RRF and why not just add the scores?"**
Cosine and BM25 live on different unnormalised scales; summing them is
meaningless. RRF discards magnitude and fuses on rank, so nothing needs calibrating.

**"Bi-encoder vs cross-encoder?"**
See Stage 5. Know this cold.

**"What would you improve?"**
Best answer: *"Contextual chunk embeddings — right now a table fragment is
embedded without any indication of which statement or period it belongs to. We
already compute page summaries, so prepending that context before embedding is
the highest-leverage fix available. Second, the evaluation set is too small to
measure whether a change helped."* Naming a specific weakness in your own system
is the strongest possible answer here.

**"How does it scale?"**
Be honest: fine to thousands of chunks per document and tens of documents. The
first bottleneck is the pure-Python BM25 (O(N) per query), then per-document
retriever loading and memory. FAISS is not the limit.

---

## 5. Study checklist

Tick these off. If you cannot explain one in two sentences, go read about it.

**Core RAG**
- [ ] What retrieval-augmented generation is and why it beats fine-tuning for document Q&A
- [ ] Chunking strategies and their trade-offs
- [ ] Embeddings, dimensionality, cosine similarity
- [ ] Why L2 normalisation makes inner product equal cosine

**Search**
- [ ] Dense vs sparse retrieval
- [ ] BM25: term frequency, IDF, length normalisation
- [ ] FAISS index types; exact vs approximate; when to switch
- [ ] Reciprocal Rank Fusion and why rank-based fusion sidesteps calibration
- [ ] Bi-encoder vs cross-encoder; the retrieve-then-rerank pattern

**LLM**
- [ ] Prompt engineering for grounding
- [ ] Temperature and why 0 for extraction
- [ ] Context windows and token budgets
- [ ] Vision-language models and multimodal input
- [ ] Hallucination: causes and mitigations

**Engineering**
- [ ] REST API design; FastAPI async vs sync handlers
- [ ] SSE vs WebSockets
- [ ] Background jobs and polling
- [ ] Token-bucket rate limiting
- [ ] Prometheus metric types and label cardinality
- [ ] Structured logging and request tracing

**This project specifically**
- [ ] Why three index tiers
- [ ] How fiscal expansion works and why it is needed
- [ ] How table chunks keep their headers
- [ ] The refusal floor and where it sits in the flow
- [ ] Citation repair
- [ ] Why FAISS was chosen, and exactly when you would replace it

---

## 6. Honesty notes

Do not overclaim. These are the places where the README is ahead of the code:

- **Intent routing affects retrieval, not generation.** `route_query` picks the
  retrieval strategy, but `answer_document_intelligence` currently ignores the
  `intent` argument it receives, so the prompt does not change per intent.
- **`route_query` returns three intents**, not four — `PRESUMMARY` is defined but
  never returned. The casual short-circuit is separate.
- **Two different refusal thresholds** exist: `chat.py` uses 0.45/1.5 and
  `intelligence.py` uses 0.30/0.60. Quote the one for the path you are describing.
- **Two metrics are declared but never incremented** (`rag_indexing_runs_total`,
  `rag_indexing_duration_seconds`), so "8 metrics" is really 6 live.

Knowing your own gaps and saying so unprompted reads as senior. Claiming
something the interviewer can disprove by opening the file does not.

See `../CODE_AUDIT.md` for the full defect list and fix order.
