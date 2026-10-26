# The Ingestion Pipeline — from PDF upload to searchable index

Everything that happens between "user drops a PDF" and "document status = ready".

This is the expensive half of the system. It runs **once per document** and takes
minutes; the query side runs constantly and takes seconds. Almost every design
decision here follows from pushing work into this pipeline so queries stay cheap.

---

## The whole pipeline at a glance

```
  PDF bytes
      │
  [0] service.ingest_pdf()          content hash → doc_id, save, spawn thread
      │
  ┌───┴─────────────── background thread: service._do_index() ────────────────┐
  │                                                                          │
  │  [1] ingest.extract_pdf()        PDF  → Page objects (text/tables/images) │
  │  [2] vision.describe_batch()     images → markdown captions               │
  │  [3] chunker.chunk_pdf()         Pages → Chunk objects                    │
  │  [4] enrichment.enrich_chunks()  Chunks → + metrics, table relationships   │
  │  [5] summarizer.build_*()        Pages → page + document summaries        │
  │  [6] HierarchicalRetriever.build()  all three tiers → embeddings          │
  │  [7] .save()                     FAISS + BM25 + JSON to disk              │
  │  [8] meta.status = "ready"                                                │
  └──────────────────────────────────────────────────────────────────────────┘
```

Each stage writes `meta.json` before it starts, so the polling UI can show live
progress: `extracting → describing images → chunking → summarizing → embedding → ready`.

---

## Stage 0 — Upload and identity

**File:** `src/service.py` → `ingest_pdf(name, pdf_bytes)`

```python
h = hashlib.sha1(pdf_bytes).hexdigest()[:12]
doc_id = f"{Path(name).stem.replace(' ', '_')[:40]}-{h}"
```

The document id is **filename + first 12 hex of the SHA1 of the file bytes** —
content addressing. Three consequences:

- Re-uploading an identical PDF resolves to the same `doc_id`, so it is a no-op
  instead of a second index.
- Two different files that happen to share a name still get distinct directories.
- The id is deterministic, so it can be recomputed anywhere without a lookup table.

The PDF is written to `storage/<doc_id>/source.pdf`, a `meta.json` is created with
`status="pending"`, and a **daemon thread** is started:

```python
t = threading.Thread(target=_do_index, args=(doc_id,), daemon=True)
t.start()
return meta          # upload returns immediately
```

Indexing a 40-page deck takes minutes. Blocking the HTTP request for that long
would time out, so the upload returns straight away and the client polls
`GET /api/documents/<id>` until `status == "ready"`.

> **Known limitation:** there is no queue or worker pool. N simultaneous uploads
> spawn N threads, each running a full embedding pass.

---

## Stage 1 — Extraction: PDF → `Page`

**File:** `src/ingest.py` → `extract_pdf(pdf_path, with_images, doc_id)`

A PDF is not a document. It is a set of glyphs with x/y coordinates. There is no
reading order, no "this is a heading", no table structure. This stage recovers
what it can with **PyMuPDF (`fitz`)**.

For every page it produces one `Page` object:

| Field | How it is obtained |
|---|---|
| `text` | `page.get_text("text", sort=True)` — `sort=True` restores reading order |
| `tables` | `page.find_tables()` → markdown, headers merged |
| `images` | `page.get_images(full=True)` + `doc.extract_image(xref)` |
| `layout` | `page.get_text("dict", sort=True)` — spans with bounding boxes |
| `section` | `_guess_section(text, previous)` |

### Three details worth knowing

**`_merge_multirow_header()`** — financial tables stack their headers:

```
| Particulars | Quarter Ended 30-09-2025 | Half Year Ended 30-09-2025 |
|             | Unaudited                | Unaudited                  |
```

Both rows are one logical header. The merger folds up to 3 rows together, so a
column label carries **both** the period and the qualifier. Losing the second row
means losing "is this audited, and is it the quarter or the half year".

**`_guess_section()`** — section headings only appear on the page where a section
*starts*. The function carries the last detected heading forward:

```python
current_section = ""
for page in doc:
    current_section = _guess_section(text, current_section)   # feeds itself
```

`return previous` is not a failure path — it is the propagation mechanism. Page 8
detects "Financial Highlights", pages 9 and 10 inherit it, page 11 replaces it.

**`table_reconstruct.py`** — when `find_tables()` fails on a complex financial
statement, this rebuilds the table geometrically: group spans into rows by
y-proximity, cluster x-positions into columns, propagate period headers across.
It is the last resort, and the hardest code in the repo.

---

## Stage 2 — Vision: images → text

**File:** `src/vision.py` → `describe_batch(images, max_workers=4, on_progress=...)`

A bar chart is invisible to a text-only pipeline — there are no words in it to
extract. This stage converts every meaningful image into searchable text.

**First, filter.** `is_meaningful_image(width, height, n_bytes)`:

```python
if width < 80 or height < 80:   return False   # icon / logo
if aspect > 12:                 return False   # divider bar, ribbon
if width * height < 8_000:      return False   # too small to hold information
if n_bytes < 1500:              return False   # glyph or flat colour fill
```

A company logo appears on all 40 pages of a deck. Without this filter you pay for
40 identical vision calls describing a logo.

**Then caption.** Survivors go to a vision LLM (`llama-3.2-11b-vision-instruct`),
base64-inlined, **4 in parallel**, with a prompt that demands a markdown table of
every data point, exact numbers with units, and the literal reply `SKIP` for
anything decorative.

```
CHART TYPE: bar
TITLE: Revenue by Segment
| Segment  | Q1 FY26 | Q2 FY26 |
|----------|---------|---------|
| Airports | ...     | ...     |
KEY TREND: Airports grew quarter-on-quarter while IRM declined.
```

Image bytes are also persisted to `storage/<doc_id>/images/p001_i00.png` so the
API can serve them back to the UI later.

**Why at index time?** Two reasons. Cost is paid once instead of per query. And
more fundamentally — at query time you would have to *find* the right chart before
you could caption it, but finding it is the very problem captioning solves. It is
circular. Captioning everything up front breaks the circle.

---

## Stage 3 — Chunking: `Page` → `Chunk`

**File:** `src/chunker.py` → `chunk_pdf(pdf_path, pages, image_captions, image_filenames)`

Two parsers, one preferred:

```python
try:
    doc = _load_opendataloader_json(pdf_path)     # primary
    return _chunks_from_opendataloader(...)
except Exception as exc:
    print(f"OpenDataLoader chunking failed; falling back to PyMuPDF: {exc}")
    return chunk_pages(...)                       # fallback
```

**OpenDataLoader** returns JSON where every element is already typed, so the
chunker routes on that type rather than guessing:

```python
kind = (element.get("type") or "").lower()

_ODL_TEXT_TYPES = {"paragraph", "heading", "caption",
                   "list", "list item", "text block"}   → text chunk
                                          "table"        → table chunk
_ODL_SKIP_TYPES = {"header", "footer", "image"}          → dropped as furniture
```

### Text and tables are chunked differently — this is the core idea

| | Text | Tables |
|---|---|---|
| Function | `_pack()` | `_split_table_markdown()` |
| Size | 900 chars (~220 tokens) | atomic; split only if forced |
| Overlap | 150 chars | none |
| Boundary safety | overlap catches split sentences | **header replayed into every piece** |

`_split_table_markdown()` is the single most important function in the ingestion
half. It locates the `|---|---|` separator line — **searching for it rather than
assuming line 1**, because the header block may be several merged rows — and then
rebuilds every piece starting from that full header:

```python
header = lines[: sep_idx + 1]
buf = header[:]
for line in lines[sep_idx + 1:]:
    if too_long:
        pieces.append("\n".join(buf))
        buf = header + [line]      # new piece starts with the header again
```

Without this, splitting a table produces `| 469.80 | 349.63 |` with nothing
saying that is finance cost for the quarter ended 30-09-2025. The number survives;
its meaning does not. This is the dominant source of wrong answers in financial RAG.

Image captions from stage 2 become their own chunks via `_append_image_chunks()`,
and `_fingerprint()` drops anything duplicated across the two parsers.

### The `Chunk` object

```python
chunk_id: int          # 0-indexed, unique within the document
page: int              # 1-indexed source page
text: str              # what actually gets embedded
kind: str              # "text" | "table" | "image"   — how it was produced
content_type: str      # refined by stage 4
doc_id, section, image_file
metrics: List[str]     # filled by stage 4
relationships: List[str]
```

`citation` is a property, and it is what the whole grounding story rests on:

```python
f"[{doc_id}:p{page}:c{chunk_id}]"    →    [Sample_-_2:p18:c110]
```

**`CHUNKER_VERSION`** is stored in `meta.json`. Bump it when chunking logic
changes and every existing index is treated as stale and rebuilt — otherwise you
silently serve queries against indexes built by old code.

---

## Stage 4 — Enrichment: extracting facts

**File:** `src/enrichment.py` → `enrich_chunks(chunks, pages)`

Pure regex and string work — **no model calls**. Deliberate: this runs over every
chunk of every document, so anything billed per token would dominate indexing
cost, and anything sampled would make indexing non-reproducible.

**The important one — `extract_table_relationships()`.** A markdown row is nearly
useless to an embedding:

```
| Finance cost | 469.80 | 349.63 |     ← almost no semantic signal
```

It is flattened into explicit facts:

```
Finance cost | Quarter Ended 30-09-2025: 469.80
Finance cost | Half Year Ended 30-09-2025: 349.63
```

This restores the binding between a number, the metric it measures and the period
it belongs to. The system prompt in `llm.py` refers to these by name and tells the
model to use them verbatim — the two files are co-designed and must change together.

Also produced: `extract_text_metrics()` (metric-bearing sentences),
`extract_graph_interpretations()` (trend language), and `detect_content_type()`,
which refines `kind` into `content_type` and adds `mixed` for prose containing an
embedded table.

---

## Stage 5 — Hierarchical summaries

**File:** `src/summarizer.py`

Builds the second and third retrieval tiers, **without an LLM**:

```python
page_summaries = build_page_summaries(pages, chunks)
doc_summary    = build_document_summary(doc_id, pages, page_summaries)
save_summaries(storage/<doc_id>/summaries.json, ...)
page_chunks, doc_chunks = summaries_to_chunks(page_summaries, doc_summary)
```

`_ranked_insights()` scores each sentence on what makes it useful as evidence —
business vocabulary (+3), contains digits (+2), movement words like "growth" or
"decline" (+2), very long (−1) — then keeps the best. It is **extractive**:
sentences are copied, never rewritten, so a summary can never introduce a fact the
page does not contain.

`build_document_summary()` splits the rollup into separate facets — highlights,
segments, trends, risks — so a question about risk does not have to out-rank every
other sentence in the report.

`summaries_to_chunks()` wraps them as ordinary `Chunk` objects using high id
offsets (`100000` for pages, `200000` for the document) so summary ids can never
collide with real chunk ids in the citation namespace.

**Why this is free:** deterministic, so it costs nothing per token, is fully
reproducible, and is cached to `summaries.json` for reuse at query time —
`intelligence._load_precomputed_summary()` reads it instead of paying for a Qwen call.

---

## Stage 6 — Embedding and index construction

**File:** `src/hierarchical.py` → `HierarchicalRetriever.build()`, then
`src/retriever.py` → `HybridRetriever.build()`

Three tiers are built, each a **full dense + sparse pair**:

```python
chunk_index    = HybridRetriever.build(chunks)                  # raw content
page_index     = HybridRetriever.build(page_summary_chunks)     # page context
document_index = HybridRetriever.build(document_summary_chunks) # narrative
```

Inside each:

```python
texts = [c.text for c in chunks]
vecs  = embed_documents(texts)        # → (N, 1024) float32, L2-normalised
index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)
bm25  = BM25Okapi([_tokenize(t) for t in texts])
```

**Embeddings** (`src/embeddings.py`) go out in batches of 32 (NVIDIA) or 50
(Gemini), with `_pace()` enforcing a minimum gap and `_maybe_429_wait()` parsing
the provider's retry hint out of the error body. Documents are embedded with
`input_type="passage"` — asymmetric embedding, which the E5 family requires; using
query mode for both quietly costs recall.

**L2 normalisation** is what makes `IndexFlatIP` a cosine index: for unit vectors,
inner product *is* cosine similarity. That is also what makes `dense_score` a
bounded `[-1, 1]` number the refusal floor can compare against a fixed threshold.

**`IndexFlatIP` = no index structure at all.** Flat means a plain float32 matrix
and exhaustive search — every query compared against every vector, 100% recall by
construction. Measured at 0.016 ms for 119 vectors, against a ~900 ms embedding
round trip. Approximate indexes (HNSW, IVF) exist to trade recall for speed at
millions of vectors; here there is no speed to buy.

---

## Stage 7 — What lands on disk

Real output for a 4-page document:

```
storage/Sample_-_2-aeef5907d9b1/
    source.pdf              184K   the original upload
    meta.json               1.0K   the record: status, counts, provenance
    faiss.index             112K   CHUNK tier — 27 x 1024 float32
    bm25.pkl                 48K   CHUNK tier — sparse index
    chunks.json              76K   CHUNK tier — text + metadata
    page_index/
        faiss.index          20K   PAGE tier
        bm25.pkl             16K
        chunks.json          28K
    document_index/
        faiss.index         8.0K   DOCUMENT tier
        bm25.pkl            8.0K
        chunks.json          12K
    summaries.json           24K   reusable page + document summaries
    suggestions.json        1.0K   LLM-generated starter questions
    images/
        p001_i00.png         12K   extracted page images
        p002_i00.png         12K
        ...
```

### The critical invariant

Within each tier, three files are **positionally aligned**:

```
FAISS row i   ==   chunks.json entry i   ==   BM25 document i
```

FAISS stores **only vectors**. `index.search()` returns row numbers — row 47 means
nothing until `chunks.json` turns it into text, page number and citation. Nothing
in the code enforces this alignment, which is why all three must always be written
together. A partial write produces an index that loads successfully and returns
the wrong text.

Index size is exactly `ntotal × dim × 4` bytes plus a ~45-byte header:
`27 × 1024 × 4 = 110,592` ≈ the 112K above.

### `meta.json`

```json
{
  "id": "Sample_-_2-aeef5907d9b1",
  "name": "Sample - 2.pdf",
  "pages": 4, "chunks": 27, "table_chunks": 13, "image_chunks": 2,
  "page_summaries": 4, "document_summaries": 1,
  "status": "ready", "stage": "", "error": null,
  "backend": "nvidia", "model": "nvidia/nv-embedqa-e5-v5",
  "chunker": "opendataloader-pdf-json-v2",
  "created_at": 1786731931.394536
}
```

`backend`, `model` and `chunker` are provenance: they let the system detect a
document indexed under different settings and rebuild it, instead of silently
mixing incompatible vectors into one query.

**The filesystem is the database.** There is no metadata store — a document exists
because its directory exists. That keeps the whole system copyable and inspectable
with nothing running, at the cost of having no transactions, which is exactly why
`status` is a state machine rather than a boolean.

---

## Failure handling

`_do_index` runs on a background thread, so it must never raise:

```python
except Exception as e:
    meta = _read_meta(doc_id) or meta
    meta.status = "error"
    meta.error  = str(e)[:500]
    _write_meta(meta)
```

Without this a failed document would sit at `indexing` forever with no explanation.
Each stage degrades independently:

| Failure | Behaviour |
|---|---|
| OpenDataLoader unavailable | falls back to PyMuPDF chunking |
| `find_tables()` misses a table | `table_reconstruct.py` rebuilds it geometrically |
| Vision backend down / `INDEX_IMAGES=false` | images skipped, text still indexed |
| A single image fails to caption | that image skipped, run continues |
| Embedding rate-limited (429) | backs off using the provider's retry hint |
| Anything else | `status="error"`, message stored in `meta.json` |

---

## Known limitations

- **`_write_meta()` is not atomic.** A crash mid-write leaves a truncated file;
  `_read_meta()` swallows the parse error and the document silently vanishes from
  listings. Fix: write to a temp file and `os.replace()`.
- **`_indexing_lock` is declared but never acquired.** Concurrent uploads run
  unbounded parallel embedding passes, and `embeddings._pace()` throttles via a
  module global with no lock, so those runs defeat the pacing they rely on.
- **`HierarchicalRetriever.index_exists()` requires all three tiers**, while
  `build()`/`load()` tolerate missing summary tiers. A document that produced no
  summaries re-indexes on every upload while remaining perfectly queryable.
- **Chunks are embedded without document context.** A table fragment carries
  `section` as metadata and shows it to the LLM, but `retriever.build()` embeds
  `c.text` only — the section never reaches the embedder. Prepending it is the
  highest-value retrieval improvement still available, and stage 1 already
  computes it.

---

## The 60-second summary

> Upload is content-addressed by SHA1 and returns immediately; indexing runs on a
> background thread with a status field the UI polls. PyMuPDF extracts text,
> tables and images with reading order restored and multi-row headers merged. A
> vision model captions charts into markdown at index time so they become
> searchable text. OpenDataLoader chunks the document by semantic element type,
> with tables kept atomic and their header row replayed into every split.
> Enrichment flattens table rows into explicit `row | column: value` facts using
> pure regex, so it is free and reproducible. Deterministic extractive summaries
> build the page and document tiers. Then all three tiers are embedded with
> asymmetric passage encoding, L2-normalised so FAISS `IndexFlatIP` behaves as an
> exact cosine index, and written to disk as three positionally-aligned files per
> tier — vectors, sparse index, and the chunk texts they point at.
