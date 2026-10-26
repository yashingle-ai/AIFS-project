<p align="center">
  <h1 align="center"> PS2 QuantX — Financial Document Intelligence RAG</h1>
  <p align="center"><em>Production-grade hierarchical Retrieval-Augmented Generation for complex financial reports</em></p>
</p>

---

##  Problem Statement

Financial documents — earnings decks, 10-K filings, annual reports — routinely span **100–1,000+ pages** with dense tables, charts, and mixed-format content. Standard RAG approaches fail because:

| Challenge | Why It Breaks Standard RAG |
|---|---|
| **Tables & Numbers** | Flat text chunking shatters tabular structures, causing hallucinated or missing financial metrics |
| **Charts & Graphs** | Text-only pipelines are **completely blind** to visual data (bar charts, waterfall diagrams, pie charts) |
| **Long-Range Context** | Small chunks lose the global narrative — "summarize this report" returns fragments, not insights |
| **Keyword vs. Semantic** | "Q2 FY26 Revenue" needs exact keyword match **and** semantic understanding of "second quarter" |
| **Hallucination** | In finance, inventing a single number can be catastrophic — zero tolerance for unsupported claims |
| **Indian Fiscal Calendar** | Users ask "Q2 FY26" but PDFs say "quarter ended 30-09-2025" — mismatch causes retrieval failure |

**PS2 QuantX solves all six problems** with a multi-layered architecture that combines layout-aware parsing, multimodal vision, hierarchical indexing, hybrid search, and strict grounding with zero-hallucination guarantees.

---

##  Our Solution — Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                │
│              "What was the EBITDA growth in Q2 FY26?"           │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                    ┌─────▼─────┐
                    │  REWRITER  │  Expand Q2 FY26 → "quarter ended
                    │ + FISCAL   │  30-09-2025" for retrieval
                    │  EXPANDER  │
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │  INTENT    │  Classify → FACTUAL | SUMMARY |
                    │  ROUTER    │  COMPARISON | PRESUMMARY
                    └─────┬─────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  CHUNK   │   │   PAGE   │   │ DOCUMENT │   ← 3-tier hierarchical
    │  INDEX   │   │  INDEX   │   │  INDEX   │     FAISS + BM25 each
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         └───────────┬──┘──────────────┘
                     ▼
              ┌─────────────┐
              │ RECIPROCAL  │  Fuse dense + sparse rankings
              │ RANK FUSION │
              └──────┬──────┘
                     ▼
              ┌─────────────┐
              │  NVIDIA     │  Cross-encoder reranking
              │  RERANKER   │  (llama-3.2-nv-rerankqa)
              └──────┬──────┘
                     ▼
              ┌─────────────┐
              │  QWEN 3-4B  │  Pre-summarize / condense
              │  SUMMARIZER │  context (optional path)
              └──────┬──────┘
                     ▼
              ┌─────────────┐
              │  PRIMARY    │  Gemini 2.5 Flash / Llama-3.3-70B
              │  LLM        │  → Grounded answer with [citations]
              └──────┬──────┘
                     ▼
              ┌─────────────┐
              │  CITATION   │  Repair, validate, append evidence
              │  ENGINE     │
              └─────────────┘
```

---

##  What We Implemented (Module by Module)

### 1. Layout-Aware PDF Ingestion (`src/ingest.py`)

We use **PyMuPDF** for page-level text, table, and image extraction with structural metadata preservation:

- **Page-aware text extraction** with reading-order normalization
- **Native table detection** via `page.find_tables()` → converted to **Markdown tables** preserving row/column structure
- **Image extraction** with intelligent filtering (`is_meaningful_image`) — skips logos, decorative bars, pixel-thin separators based on dimensions, aspect ratio, and byte size
- **Section detection** — heuristic heading identification to tag chunks with their section context
- **Layout block extraction** — extracts bounding-box metadata from PyMuPDF's structured dict output

### 2. Semantic Chunking with OpenDataLoader (`src/chunker.py`)

**Primary parser: OpenDataLoader** (structured PDF→JSON conversion) with PyMuPDF fallback:

- **OpenDataLoader integration** — uses `opendataloader_pdf.convert()` with `table_method="cluster"` and `reading_order="xycut"` to preserve reading order and table boundaries
- **Element-type-aware processing** — handles paragraphs, headings, captions, lists, tables, and list items differently; skips headers/footers
- **Tables as atomic chunks** — tables are never split mid-row; large tables are intelligently split while preserving the header row in each piece
- **Paragraph packing with overlap** — text paragraphs are packed into ~900-character chunks with 150-char overlap for context continuity
- **Image chunks** — vision LLM captions become searchable chunks tagged as `kind="image"` or `kind="table"` (if the caption contains extracted tabular data)
- **Deduplication** — fingerprint-based deduplication prevents duplicate tables from OpenDataLoader and PyMuPDF

### 3. Chunk Enrichment (`src/enrichment.py`)

Deterministic, pre-embedding enrichment that makes chunks more retrievable:

- **Table enrichment** — parses Markdown tables into structured `row | column: value` relationship facts (e.g., `Revenue / Q2 FY26: ₹4,523 Cr`)
- **Metric extraction** — regex-based extraction of financial values with their surrounding sentence context
- **Graph/chart interpretation** — identifies trend-bearing sentences referencing charts, growth, decline
- **Surrounding context injection** — table chunks are enriched with nearby metric-bearing sentences from the same page
- **Content type classification** — chunks are tagged as `text`, `table`, `mixed`, or `image`

### 4. Vision Pipeline — Chart-to-Text (`src/vision.py`)

Makes visual content **searchable** by converting images to structured text descriptions at **index time**:

- **Dual backend**: NVIDIA (`meta/llama-3.2-11b-vision-instruct`) or Google Gemini (`gemini-2.5-flash`)
- **Financial-domain prompt engineering** — extracts charts into Markdown tables with exact data points, axis labels, legends, and trend descriptions
- **Smart image filtering** — skips images < 80×80px, extreme aspect ratios (>12:1), tiny byte sizes, ensuring only meaningful charts/graphs are processed
- **Concurrent batch captioning** — `ThreadPoolExecutor` with 4 workers for parallel image processing with live progress reporting
- **Graceful failure** — vision failures never break the indexing pipeline; images degrade to no-caption silently
- **One-time cost** — captioning happens during indexing, not per-query, so runtime latency is unaffected

### 5. Hierarchical Summarization (`src/summarizer.py`)

Pre-computed, deterministic summaries at page and document levels:

- **Page summaries** include: key insights (ranked by business-relevance scoring), important metrics, and data interpretations
- **Document summary** aggregates all page summaries into: key highlights, financial/business metrics, segment analysis, trends, and risks
- **Business-aware sentence ranking** — sentences are scored by presence of business keywords, numeric values, and trend indicators
- **Zero LLM cost** — summarization is purely extractive (regex + heuristics), not generative, so it requires no API calls and is deterministic

### 6. Qwen Summarizer — Second LLM Path (`src/qwen_summarizer.py`)

A **dedicated lightweight model** (Qwen3-4B-Instruct via HuggingFace Inference API) for intelligent summarization:

- **Three modes**:
  - **Page summary** — user asks "summarize page 5" → Qwen generates a structured summary
  - **Document summary** — user asks "give me an overview" → Qwen generates an executive summary
  - **Pre-summary** — complex analytical queries → Qwen condenses retrieved context by 40-60% before the primary LLM processes it
- **Intent detection** — regex-based classification of whether a query needs the Qwen path
- **Citation preservation** — Qwen summaries retain `[doc:pX:cY]` citations for the primary LLM to verify against

### 7. Hybrid Retrieval — FAISS + BM25 (`src/retriever.py`)

Dual-signal retrieval with **Reciprocal Rank Fusion**:

- **FAISS IndexFlatIP** — inner-product search on L2-normalized vectors (= cosine similarity)
- **BM25Okapi** — exact keyword matching using `rank_bm25` with financial-token-aware tokenization
- **RRF fusion** — `score = Σ 1/(k + rank)` across dense and sparse rankings, combining semantic and lexical signals
- **Persistence** — indexes saved as `faiss.index` + `bm25.pkl` + `chunks.json`; zero reindexing on restart

### 8. Multi-Backend Embeddings (`src/embeddings.py`)

Hot-swappable embedding backends:

| Backend | Model | Batch Size | Rate Pacing |
|---|---|---|---|
| **NVIDIA NIM** | `nvidia/nv-embedqa-e5-v5` | 32 | 0.1s min gap |
| **Google Gemini** | `gemini-embedding-001` | 50 | 0.3s min gap |

- L2 normalization for consistent cosine similarity scoring
- Automatic 429/rate-limit detection with exponential backoff and `retryDelay` header parsing
- Progress reporting for large document embedding

### 9. Multi-Backend LLM (`src/llm.py`)

Swappable generation backends:

| Backend | Model | Use |
|---|---|---|
| **NVIDIA NIM** | `meta/llama-3.3-70b-instruct` | Primary chat/generation |
| **Google Gemini** | `gemini-2.5-flash` | Primary chat/generation |

- **Advanced 6-step system prompt** — dynamically adapts response strategy based on query intent (fact lookup, summarization, analysis, comparison, table query)
- **Conversational rewriter** — resolves pronouns and references from chat history into standalone retrieval queries
- **Strict grounding rules** — "Use ONLY the provided context. Do NOT hallucinate."
- **Multi-document citation format** — `[doc1:p12:c3]`, `[doc2:p5:c1]`

### 10. Hierarchical Multi-Document Retriever (`src/hierarchical.py`)

Three-tier index architecture per document:

```
HierarchicalRetriever
├── chunk_index:    HybridRetriever  ← raw text/table/image chunks
├── page_index:     HybridRetriever  ← page-level summary chunks
└── document_index: HybridRetriever  ← document-level summary chunks
```

- **MultiDocumentRetriever** — wraps multiple `HierarchicalRetriever` instances for cross-document querying
- **Level-aware search** — `search_levels()` queries across specified tiers and merges results
- **Per-document filtering** — `filtered(doc_ids)` scopes queries to a document subset
- **Deduplication** — citation-based dedup prevents redundant hits across levels

### 11. Intelligence Layer — Intent Routing (`src/intelligence.py`)

The brain of the system — `DocumentIntelligenceRAG`:

- **Intent classification**: `FACTUAL`, `SUMMARY`, `COMPARISON`, `PRESUMMARY` via regex patterns
- **Dynamic retrieval strategy**:
  - `FACTUAL` → chunk-level search
  - `SUMMARY` → page + chunk levels (or document + page + chunk if "full report")
  - `COMPARISON` → all three levels with wider top-k
- **Adaptive top-k** — broad queries automatically widen to k=12+ for completeness
- **Refusal floor** — refuses if best `dense_score < 0.30` **and** best `sparse_score < 0.60`
- **Citation repair** — fixes malformed citations by matching page/chunk IDs against actual retrieved hits
- **Extractive fallback** — if the LLM backend is unavailable, returns raw evidence with citations instead of crashing

### 12. Conversational RAG (`src/chat.py`)

Single-document conversational chain with domain-specific features:

- **Indian fiscal calendar expansion** — automatically expands `Q2 FY26` → `quarter ended 30-09-2025`, `H1 FY26` → `half year ended 30-09-2025`, and bare `FY26` → `year ended 31-03-2026`
- **Broad question detection** — "summarize", "list all", "outline" triggers widened recall (k=12)
- **Cross-encoder reranking integration** — when enabled, retrieves 30 candidates and reranks to top-k

### 13. Cross-Encoder Reranking (`src/rerank.py`)

Optional NVIDIA-hosted reranker for precision:

- **Model**: `nvidia/llama-3.2-nv-rerankqa-1b-v2`
- Fetches wide candidate set → reranks by cross-encoder logit → trims to top-k
- **Silent fallback** — reranker failures never break the pipeline; falls back to original RRF ordering

### 14. FastAPI Server (`server.py`)

Production-grade REST API with 12+ endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/info` | GET | Active backends and models |
| `/api/metrics` | GET | Prometheus exposition format |
| `/api/documents` | GET/POST | List or upload documents |
| `/api/documents/{id}` | GET/DELETE | Get status or remove |
| `/api/documents/{id}/pdf` | GET | Serve original PDF |
| `/api/documents/{id}/images/{name}` | GET | Serve extracted images |
| `/api/documents/{id}/suggestions` | GET | LLM-generated starter questions |
| `/api/documents/{id}/summaries` | GET | Pre-computed hierarchical summaries |
| `/api/documents/{id}/chat` | POST | Blocking Q&A |
| `/api/documents/{id}/chat/stream` | POST | SSE streaming Q&A |
| `/api/chat` | POST | Multi-document cross-query |
| `/api/feedback` | POST | Thumbs up/down with JSONL logging |

- **Background indexing** — uploads return immediately; indexing runs in a daemon thread with live progress via polling
- **SSE streaming** — server-sent events with `meta → token → done` frame protocol
- **Cache layer** — thread-safe lazy retriever cache per document

### 15. Security Hardening (`src/security.py`)

Six-layer security stack:

1. **CORS** — restricted to localhost dev ports; configurable via `ALLOW_ORIGINS`
2. **CSP headers** — Content-Security-Policy, X-Frame-Options: DENY, no-sniff, no-referrer
3. **Bearer token auth** — opt-in via `RAG_AUTH_TOKEN` env; constant-time comparison via `hmac.compare_digest`
4. **Token-bucket rate limiter** — per-IP with separate buckets for upload (0.2 rps) vs. default (5 rps)
5. **Path traversal prevention** — `validate_doc_id` rejects `..`, leading/trailing dots, and non-alphanumeric chars
6. **Input validation** — question length (2,000 chars), history turns (20 max), history size (32KB max)

### 16. Observability (`src/observability.py`)

Production monitoring:

- **Structured JSON logging** — every request logged with request_id, route, status, latency_ms, client IP, user agent
- **Request-scoped context** — `ContextVar` propagates request_id through all log calls
- **Prometheus metrics** — 8 named metrics including:
  - `rag_http_requests_total` (method, route, status)
  - `rag_http_request_duration_seconds` (histogram with 9 buckets)
  - `rag_chat_requests_total` (backend, model, refused)
  - `rag_chat_tokens_total` (prompt/completion direction)
  - `rag_indexing_runs_total` + `rag_indexing_duration_seconds`
  - `rag_documents_indexed` (gauge)
  - `rag_retrieval_top_dense_score` (histogram)
  - `rag_feedback_total` (up/down)

### 17. Document Service (`src/service.py`)

Backend orchestration and document lifecycle:

- **Indexing pipeline**: extract → vision caption → chunk → enrich → summarize → embed → save
- **Per-document storage** with `meta.json` tracking: status (pending → indexing → embedding → ready), chunk counts, backend info, errors
- **Dedup on re-upload** — SHA1 content hash prevents re-indexing identical PDFs
- **Chunker versioning** — `CHUNKER_VERSION` ensures stale indexes are rebuilt when parsing logic changes

### 18. Evaluation Harness (`tests/eval.py` + `tests/acceptance.py`)

Rigorous quality measurement:

- **Golden dataset** — JSONL format with `q`, `expected_pages`, `must_contain`, `must_contain_any`, `refuse` fields
- **4 metrics**: Recall@k, Page Precision, Refusal Accuracy, Faithfulness
- **Acceptance tests** — 5 must-pass scenarios: grounded fact, numeric extraction, cross-section, negative control (refusal), multi-turn follow-up

---

##  Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11+ |
| **API Framework** | FastAPI + Uvicorn |
| **Web UI** | Streamlit (app.py), React SPA (static/index.html, no build step) |
| **PDF Parsing** | OpenDataLoader (primary), PyMuPDF (fallback + images) |
| **Dense Index** | FAISS (IndexFlatIP, cosine via inner product) |
| **Sparse Index** | BM25Okapi (rank_bm25) |
| **Embeddings** | NVIDIA NV-EmbedQA-E5-v5 / Gemini Embedding 001 |
| **Primary LLM** | Meta Llama-3.3-70B-Instruct / Gemini 2.5 Flash |
| **Vision LLM** | Meta Llama-3.2-11B-Vision-Instruct / Gemini 2.5 Flash |
| **Summarizer LLM** | Qwen3-4B-Instruct (via HuggingFace) |
| **Reranker** | NVIDIA Llama-3.2-NV-RerankQA-1B-v2 |
| **Monitoring** | Prometheus + structured JSON logs |
| **CLI** | Rich (tables, panels, markdown rendering) |

---

##  Quickstart

### 1. Install

```bash
cd rag_app
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Edit .env → add your API keys (NVIDIA_API_KEY, GEMINI_API_KEY, HF_API_KEY)
```

### 2. Choose Your Backends (`.env`)

```bash
LLM_BACKEND=nvidia          # or "gemini"
EMBED_BACKEND=nvidia         # or "gemini"
VISION_BACKEND=nvidia        # or "gemini" or "none"
RAG_PDF_PARSER=opendataloader  # or "pymupdf" or "auto"
```

### 3. Run the Web UI (Streamlit)

```bash
streamlit run app.py
# Upload a PDF → ask questions → see retrieved chunks in debug panel
```

### 4. Run the API Server (FastAPI + React SPA)

```bash
uvicorn server:app --port 8000 --reload
# Open http://localhost:8000
```

### 5. CLI — Single-shot or Interactive

```bash
# Single question
python main.py --pdf ../testing_files/Sample\ -\ 1.pdf --ask "What are the key revenue streams?"

# Multi-document interactive chat
python main.py --pdf ../testing_files/Sample\ -\ 1.pdf ../testing_files/Sample\ -\ 2.pdf

# In the chat loop:
#   /reset  — clear conversation history
#   /debug  — toggle retrieved-context visibility
#   /k 12   — change top-k
#   /quit   — exit
```

---

##  Project Structure

```
rag_app/
├── server.py                  # FastAPI REST API (12+ endpoints, SSE streaming)
├── app.py                     # Streamlit chat UI
├── main.py                    # CLI entry point (single/multi-doc, interactive)
├── requirements.txt           # Python dependencies
├── .env.example               # Configuration template
│
├── src/                       # Core modules
│   ├── ingest.py              # PDF → pages (text + tables + images)
│   ├── chunker.py             # OpenDataLoader / PyMuPDF → semantic chunks
│   ├── enrichment.py          # Table relationship extraction, metric extraction
│   ├── embeddings.py          # NVIDIA / Gemini embedding dispatch
│   ├── retriever.py           # FAISS + BM25 + RRF hybrid search
│   ├── hierarchical.py        # 3-tier indexing, multi-document retrieval
│   ├── summarizer.py          # Deterministic page/document summaries
│   ├── qwen_summarizer.py     # Qwen3-4B summarization pipeline
│   ├── vision.py              # Chart/image → text via vision LLMs
│   ├── llm.py                 # LLM dispatch (NVIDIA / Gemini), prompts
│   ├── chat.py                # Conversational RAG, fiscal expansion
│   ├── intelligence.py        # Intent routing, answer normalization
│   ├── rerank.py              # Cross-encoder reranking (NVIDIA)
│   ├── service.py             # Document registry, background indexing
│   ├── security.py            # CORS, CSP, auth, rate limiting
│   └── observability.py       # JSON logging, Prometheus metrics
│
├── static/
│   └── index.html             # React SPA (no build step, 83KB)
│
├── tests/
│   ├── eval.py                # Recall/refusal/faithfulness eval harness
│   ├── acceptance.py          # 5 must-pass acceptance tests
│   └── golden_AEL.jsonl       # Golden dataset for AEL earnings deck
│
├── storage/                   # Runtime: per-document indexes & artifacts
│   └── <doc_id>/
│       ├── source.pdf
│       ├── meta.json
│       ├── faiss.index
│       ├── bm25.pkl
│       ├── chunks.json
│       ├── summaries.json
│       ├── suggestions.json
│       ├── images/            # Extracted chart images
│       ├── page_index/        # Page-level FAISS + BM25
│       └── document_index/    # Document-level FAISS + BM25
│
└── testing_files/             # Sample PDFs for testing
    ├── AEL_Earnings_Presentation_Q2-FY26.pdf
    ├── Sample - 1.pdf
    └── Sample - 2.pdf
```

---

##  Testing & Evaluation

### Run the Eval Harness

```bash
python -m tests.eval \
    --pdf ../testing_files/AEL_Earnings_Presentation_Q2-FY26.pdf \
    --golden tests/golden_AEL.jsonl
```

**Metrics computed per question:**
- **Recall@k** — did at least one expected page appear in top-k retrieved chunks?
- **Page Precision** — fraction of retrieved pages that were expected
- **Refusal Accuracy** — did it correctly refuse off-topic queries and correctly answer on-topic ones?
- **Faithfulness** — do all `must_contain` strings appear verbatim in the answer?

### Run Acceptance Tests

```bash
python -m tests.acceptance
```

5 must-pass scenarios: grounded fact extraction, numeric precision, cross-section analysis, negative control (refusal), and multi-turn conversational follow-up.

---

##  Key Design Decisions

| Decision | Rationale |
|---|---|
| **OpenDataLoader as primary parser** | Preserves reading order, table structure, and element boundaries that PyMuPDF destroys |
| **Vision captioning at INDEX time** | One-time cost; queries see chart data as plain text with zero runtime overhead |
| **Extractive (not generative) summaries** | Deterministic, zero LLM cost, no hallucination risk in the summary layer |
| **Qwen as a second LLM** | Lightweight 4B model for summarization tasks — keeps primary LLM context window focused |
| **RRF fusion over learned fusion** | No training data needed; works out-of-the-box; tunable via `rrf_k` parameter |
| **Indian fiscal calendar expansion** | Domain-specific retrieval boost — "Q2 FY26" → "quarter ended 30-09-2025" |
| **Token-bucket rate limiter** | In-process, zero external deps; separate buckets for expensive (upload) vs. cheap (chat) routes |
| **Chunker versioning** | `CHUNKER_VERSION` forces index rebuild when parsing logic changes — no stale indexes |

---

##  Team

**PS2 QuantX** — Built for the Powermind Hackathon
YASH INGLE - TEAM LEADER 
HIMAL RANA - TEAM MEMBER 
ANKIT YADAV - TEAM MEMBER 
TARUN BHUTRA - TEAM MEMBER 
---
