# Backend Code Audit — `rag_app/src`

Scope: all 17 modules, ~5,550 LOC. Function-level review for correctness,
robustness and production-readiness.

**Overall: B+.** Architecture is sound and the error-handling discipline is
better than most production codebases. Defects are concentrated in operational
concerns (concurrency, cache lifecycle, metric labels), not in retrieval logic.

---

## Severity summary

| # | Severity | Location | Issue |
|---|---|---|---|
| 1 | **High** | `observability.py:189` | Route label reads `scope["route"]` before routing → raw path becomes the label → unbounded Prometheus cardinality |
| 2 | **High** | `service.py:257` + `hierarchical.py:66` | `index_exists` demands all 3 tiers; `get_hierarchical_retriever` demands only 1 → re-index loop on docs with no summaries |
| 3 | **High** | `security.py:161` | `X-Forwarded-For` trusted unconditionally → rate-limit bypass + unbounded bucket dict |
| 4 | **High** | `service.py:142` | `_indexing_lock` declared, never used; unbounded daemon threads; `embeddings._pace()` global is not thread-safe |
| 5 | Medium | `chat.py:107` vs `intelligence.py:24` | Two different refusal thresholds for the same guarantee (0.45/1.5 vs 0.30/0.60) |
| 6 | Medium | `llm.py:271` | `answer_document_intelligence` accepts `intent` and ignores it — identical to `answer()` |
| 7 | Medium | `service.py:81` | `_write_meta` is non-atomic → a torn write makes the document vanish from listings |
| 8 | Medium | `service.py:283` | Retriever caches never evict |
| 9 | Medium | `rerank.py:67` | Failures swallowed with no log and no metric |
| 10 | Medium | `chat.py:185` | Refusal detected by substring match on model prose |
| 11 | Low | `observability.py:149` | `indexing_runs_total` / `indexing_duration_seconds` never incremented |
| 12 | Low | `intelligence.py:21` | `PRESUMMARY` constant is dead |

---

## Cross-cutting findings

### 1. Prometheus label cardinality (High)

`RequestLogMiddleware.dispatch` reads the route **before** `call_next`, but
Starlette populates `scope["route"]` during dispatch. The `getattr` fallback
therefore always wins and the raw path becomes the label.

Verified — three requests to one endpoint produced three series:

```
/api/documents/aaa111/chat
/api/documents/bbb222/chat
/api/documents/ccc333/chat
```

Every document ID becomes a permanent time series. Fix: move both lines into
the `finally` block.

### 2. Inconsistent index-existence semantics (High)

| Caller | Predicate | Effect |
|---|---|---|
| `service.ingest_pdf:257` | `HierarchicalRetriever.index_exists` (all 3 tiers) | dedup fails |
| `service.get_hierarchical_retriever:305` | `HybridRetriever.index_exists` (chunk only) | load succeeds |

`HierarchicalRetriever.build` permits `page_index`/`document_index` to be
`None`, and `save()` skips what is `None` — but `index_exists` requires all
three on disk. A document that yields no summaries therefore re-indexes on
every upload while still being queryable. Align both call sites on the
chunk-index predicate.

### 3. Rate limiter trusts client-supplied headers (High)

```python
xff = request.headers.get("x-forwarded-for", "")
if xff:
    return xff.split(",")[0].strip()
```

Directly exposed, any client sets this freely: rotate the value and the
limiter never fires, while `self._buckets` grows without bound. Gate on a
`TRUST_PROXY` env flag and add TTL eviction.

### 4. Indexing concurrency is unbounded (High)

`ingest_pdf` spawns a daemon thread per upload with no pool or queue.
`_indexing_lock` exists but is never acquired. Meanwhile
`embeddings._pace()` mutates a module-global `_last_call_at` without a lock,
so concurrent indexing silently defeats the rate pacing it implements.

### 5. Two refusal thresholds (Medium)

| Path | Dense | Sparse |
|---|---|---|
| `chat.ConversationalRAG` | 0.45 | 1.5 |
| `intelligence.DocumentIntelligenceRAG` | 0.30 | 0.60 |

The same question refuses in one endpoint and answers in the other. Whichever
is correct, it should live in one module.

---

## Per-module review

### `retriever.py` — 120 LOC · **A−**
Cleanest module in the codebase.

| Function | Assessment |
|---|---|
| `_tokenize` | Regex keeps `-_/%.` — correct for tickers and `Q2/FY26`. No stemming or stopwords, appropriate for numeric retrieval. |
| `HybridRetriever.build` | `IndexFlatIP` over L2-normalised vectors = exact cosine. Correct and deliberate. |
| `save` / `load` | Three-file persistence. **`pickle.load` on `bm25.pkl` is arbitrary code execution if storage is ever attacker-writable** — acceptable locally, not for multi-tenant. |
| `_dense` | Guards `i != -1` for under-filled results. Correct. |
| `_sparse` | `get_scores` is O(N) over the whole corpus per query — pure Python, the real scaling bottleneck. |
| `search` | Textbook RRF (`k=60`). Ignores score magnitude by design; the raw scores are preserved on `Hit` for the refusal floor. Sound. |

### `observability.py` — 215 LOC · **B**

| Function | Assessment |
|---|---|
| `JsonFormatter.format` | Correct reserved-attr filtering; falls back to `repr` on non-serialisable values. |
| `configure_logging` | Replaces root handlers, quiets noisy libs. Correct. |
| `_SafeLogger` | Renames `extra=` keys colliding with `LogRecord` slots — prevents a real production `KeyError`. Genuinely thoughtful. |
| `metrics_response` | Fine. Endpoint sits behind auth (not in `_PUBLIC_PREFIXES`) — correct. |
| `RequestLogMiddleware.dispatch` | Metrics in `finally` (counts 5xx). `ContextVar` set/reset correct. **Finding #1 lives here.** SSE duration measures time-to-first-byte. |

Two metrics are declared and never incremented (#11).

### `security.py` — 228 LOC · **B+**

| Function | Assessment |
|---|---|
| `cors_origins` | Localhost default, env override. Correct. |
| `SecurityHeadersMiddleware` | Uses `setdefault` so handlers can override. CSP carries `'unsafe-inline' 'unsafe-eval'` for in-browser Babel — necessary, but it materially weakens the CSP claim. Say "CSP with a documented `unsafe-eval` exemption for the CDN-based SPA". |
| `BearerAuthMiddleware` | `hmac.compare_digest` — correct. Auth is opt-in; off by default. |
| `_Bucket.take` | Correct monotonic token bucket. |
| `RateLimitMiddleware` | Lock-guarded. **Findings #3.** |
| `validate_doc_id` | Three independent checks (regex, `..`, `PurePosixPath.name`). Exemplary defence in depth. |
| `safe_filename` | Strips separators then filters. Correct. |
| `validate_chat_payload` | `getattr(h, "content", "")` returns `""` for plain dicts — **verify the caller passes Pydantic models, or the size check is a silent no-op.** |

### `service.py` — 329 LOC · **B−**
Highest concentration of issues.

| Function | Assessment |
|---|---|
| `_read_meta` | Bare `except` → `None`. Combined with non-atomic writes, a torn file silently deletes the doc from listings. |
| `_write_meta` | **Not atomic (#7).** Write to `.tmp` + `os.replace`. |
| `image_path` | Regex + `resolve().relative_to()`. Correct. |
| `_do_index` | Good stage reporting and a catch-all that records `error` in meta. `progress()` re-reads meta while the outer scope holds a stale copy — lost-update race on `stage` (benign). Indexing is never timed (#11). |
| `ingest_pdf` | SHA1 content dedup is right. **Findings #2, #4.** |
| `get_retriever` / `get_hierarchical_retriever` | Lock-guarded, lazy. **No eviction (#8).** |
| `delete_document` | Pops caches **outside** `_cache_lock` — races `get_retriever`. |
| line 119 | `__import__("re")` instead of a module-level import. |

### `intelligence.py` — 652 LOC · **B**
The orchestrator. Largest file; would benefit from decomposition.

| Function | Assessment |
|---|---|
| `route_query` | Returns only `COMPARISON`/`SUMMARY`/`FACTUAL`. `PRESUMMARY` is never returned (#12) — README's "4 intents" is really 3 + a separate casual short-circuit. |
| `format_hierarchical_context` | Enforces a 12k char budget with proportional allocation to secondary hits. Good. |
| `ask` | Clear pipeline. Sorts by `dense_score` because RRF scores are per-subquery and not comparable — correct reasoning, documented in-line. `k_limit = top_k * len(sub_queries)` can inflate context on multi-part questions. |
| `_insufficient` | The real refusal floor. Correct; see #5. |
| `_handle_summary_query` | Session cache → precomputed disk summary → Qwen call. Right order (cheapest first). Nested try/except degrades cleanly. |
| `_direct_page_lookup` | Bypasses embeddings for "page 19" queries. Correct instinct — semantic search cannot resolve page numbers. |
| `_extractive_fallback` | Returns cited evidence when the LLM is unavailable rather than 500-ing. Excellent. |
| `_normalize_answer` | Citation repair against actually-retrieved hits. This is what makes the grounding claim real. |

### `chat.py` — 193 LOC · **B**

| Function | Assessment |
|---|---|
| `_normalize_fy` | `<100 → +2000`. Correct. |
| `expand_fiscal_terms` | The domain differentiator. Correct Indian FY mapping (Q4 uses offset `0`, Q1–Q3 use `-1`). Appends to the retrieval query only, leaving the user's question intact — the right call. |
| `_effective_top_k` | Widens k for broad questions. Sound. |
| `format_context` | Truncates at 1,800 chars/chunk. Fine. |
| `ConversationalRAG.ask` | **#5** and **#10**: `if "not found in the document" in ans_text.lower()` is brittle — a differently-worded refusal is missed, and a document legitimately containing that phrase triggers a false refusal. Compare against the constant, or have the model emit a structured marker. |

### `llm.py` — 290 LOC · **B−**

| Function | Assessment |
|---|---|
| `ADVANCED_SYSTEM` | Strong prompt. The TABLE_QUERY row/column rules directly target the dominant financial-RAG failure mode. |
| `_maybe_wait_429` | Parses `retryDelay`. Good. Retries on non-retryable errors (401) too — wasteful, not harmful. |
| `_nvidia_generate` | `max_tokens=2048`. Correct retry/raise structure. |
| `_gemini_generate` | **No output cap** — asymmetric with the NVIDIA path. |
| `decompose_and_rewrite` | Regex fallback when the rewriter fails. Good. |
| `answer` | Correct. |
| `answer_document_intelligence` | **#6 — byte-identical to `answer()` plus an ignored `intent` parameter.** Either thread `intent` into the prompt or delete the function. The README's "dynamically adapts response strategy based on query intent" is not true at generation time; intent only affects *retrieval*. |
| `NOT_FOUND` / `NOT_FOUND_DOCUMENTS` | Identical strings; collapse to one. |

### `embeddings.py` — 196 LOC · **B+**

| Function | Assessment |
|---|---|
| `_normalize` | Guards zero-norm rows. Correct — this is what makes `IndexFlatIP` a cosine index. |
| `_pace` | Module-global, **not thread-safe (#4)**. |
| `_maybe_429_wait` | Correct. |
| `_nvidia_embed` / `_gemini_embed` | Correct `for/else` retry idiom. Uses `input_type=passage/query` — asymmetric embedding, which is right for E5-family models and often missed. |
| `embed_documents` | Empty case returns width `_known_dim or 1024` — a wrong guess would produce a dimension-mismatched index. |
| progress output | Uses `print()` rather than the structured logger. |

### `hierarchical.py` — 151 LOC · **B**

| Function | Assessment |
|---|---|
| `build` / `save` / `load` | Tolerate missing tiers consistently. |
| `index_exists` | **#2** — inconsistent with the above. |
| `MultiDocumentRetriever.search` | Loops every retriever: O(documents) per query. |
| `_dedupe_hits` | Citation-keyed. Correct. |
| `_global_score` | Sums cosine (0–1), RRF (~0.02) and sparse/100 — incomparable scales, so ranking is effectively dense-dominated. Works, but arbitrary; bypassed entirely when reranking is on. |

### `rerank.py` — 69 LOC · **B**

| Function | Assessment |
|---|---|
| `_nvidia_rerank` | Truncates passages to 4,000 chars for the NIM cap. Correct. |
| `rerank` | Never breaks the pipeline — right policy. **#9: the `except` is silent.** A bad API key degrades precision permanently with zero signal. Log a warning and add a `rag_rerank_failures_total` counter. |

### `ingest.py` — 261 LOC · **A−**

| Function | Assessment |
|---|---|
| `_merge_multirow_header` | Merges up to 3 header rows — essential for financial statements with stacked period headers. Strong. |
| `_table_to_markdown` | Handles `None` cells. |
| `_extract_images` | Filters via `is_meaningful_image` before captioning — avoids paying for logos and rules. |
| `extract_pdf` | Clean PyMuPDF fallback path. |

### `chunker.py` — 625 LOC · **A−**
The strongest domain work in the repo.

| Function | Assessment |
|---|---|
| `_pack` | Character packing with overlap. Fine. |
| `_opendataloader_table_to_markdown` | Table→markdown preserving structure. |
| `_split_table_markdown` | **Replays the header row into every split** — the single most valuable function here, and exactly what prevents fabricated metrics. |
| `_is_section_heading` / `_element_page` | Reasonable heuristics. |
| `_fingerprint` | Dedup by content hash. |
| `CHUNKER_VERSION` | Invalidates stale indexes when parsing logic changes. Mature. |

### `enrichment.py` — 266 LOC · **B+**

| Function | Assessment |
|---|---|
| `extract_table_relationships` | Emits `<row> \| <column>: <value>` facts that the system prompt explicitly consumes. Prompt and enrichment are co-designed — good. |
| `extract_text_metrics` / `extract_graph_interpretations` | Deterministic, cacheable, zero API cost. |
| `detect_content_type` | Fine. |
| **Gap** | No document/section context is prepended before embedding — the highest-leverage improvement available (contextual retrieval). |

### `vision.py` — 204 LOC · **A−**

| Function | Assessment |
|---|---|
| `is_meaningful_image` | Dimension/byte thresholds filter decorative images. Direct cost control. |
| `describe_image` | Returns `Optional[str]`; `None` skips cleanly. |
| `describe_batch` | `ThreadPoolExecutor(max_workers=4)` with progress callback. Correct — indexing is I/O-bound. |

### `summarizer.py` — 225 LOC · **A−**
Fully deterministic (no LLM): ranked-sentence extraction into page and document
summaries, persisted to `summaries.json` and reused later by
`intelligence._load_precomputed_summary`. Free, reproducible, cache-friendly.

### `qwen_summarizer.py` — 240 LOC · **B+**

| Function | Assessment |
|---|---|
| `needs_summary` | Returns `"page"` / `"document"` / `"pre"` / `None` — a routing signal separate from `route_query`. Two parallel routing systems is a design smell worth consolidating. |
| `_qwen_generate` | HF-hosted, guarded. |
| `summarize_context` | Context compression before the primary LLM. Sound use of a small model. |

### `table_reconstruct.py` — 359 LOC · **A**
Geometric table reconstruction from PyMuPDF spans: row grouping by y-tolerance,
x-position clustering into columns, period propagation across merged headers.
Genuinely difficult code, and the right answer for financial statements where
text extraction destroys layout. `_propagate_periods` and `_find_data_start`
are the load-bearing pieces.

---

## What is done well

- **Async hygiene** — blocking endpoints are `def` (FastAPI threadpool); the one
  `async` streaming endpoint uses `run_in_executor`. The event loop is never
  blocked. Frequently botched; correct here.
- **Graceful degradation everywhere** — reranker, vision, Qwen and the primary
  LLM each have a fallback. The system answers with cited evidence even with
  the LLM down.
- **Path-traversal defence in depth** — three independent checks, applied in two
  modules.
- **Cost discipline** — meaningful-image filtering, deterministic summaries,
  precomputed-summary reuse, session caching.
- **Index invalidation** — `CHUNKER_VERSION` plus SHA1 content addressing.
- **Retrieval correctness** — L2-normalise → `IndexFlatIP`, asymmetric
  passage/query embedding, RRF at k=60. All textbook-correct.

## Recommended fix order

1. `observability.py:189` — move the route lookup into `finally` (2 lines)
2. `service.py:257` — align on `HybridRetriever.index_exists` (1 line)
3. `security.py:161` — gate XFF behind `TRUST_PROXY`, add bucket TTL
4. `service.py` — bound indexing concurrency; lock `_pace`
5. Unify the refusal thresholds into one module
6. Either use `intent` in `answer_document_intelligence` or delete it
7. `_write_meta` — atomic temp + `os.replace`
8. Log and count reranker failures
