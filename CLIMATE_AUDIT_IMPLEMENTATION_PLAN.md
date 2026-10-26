# GenAI-Assisted Climate Risk Auditing — Implementation Plan

How to build the climate-claim verification system described in
[GenAI-Assisted Climate Risk Auditing — Build Plan.md](GenAI-Assisted%20Climate%20Risk%20Auditing%20—%20Build%20Plan.md)
on top of the existing QuantX financial RAG codebase (`rag_app/`), as a research
project that ends in a workshop paper.

This document covers:

1. What the existing codebase gives us and what is missing
2. The target system and the research questions
3. The build, in ten phases, in the order they should be done
4. Datasets, evaluation, hallucination controls, and LangSmith
5. Timeline, risks, and the paper outline

---

## Contents

- [Part A — Analysis of the existing project](#part-a--analysis-of-the-existing-project)
- [Part B — Target system and research framing](#part-b--target-system-and-research-framing)
- [Part C — Build phases (logical sequence)](#part-c--build-phases-logical-sequence)
  - [Phase 0 — Foundations and reproducibility](#phase-0--foundations-and-reproducibility)
  - [Phase 1 — Data collection and annotation setup](#phase-1--data-collection-and-annotation-setup)
  - [Phase 2 — Ingestion upgrade (pipeline stage 1)](#phase-2--ingestion-upgrade-pipeline-stage-1)
  - [Phase 3 — In-report retrieval and the first benchmark (pipeline stage 4a)](#phase-3--in-report-retrieval-and-the-first-benchmark-pipeline-stage-4a)
  - [Phase 4 — Claim detection and structured extraction (pipeline stage 2)](#phase-4--claim-detection-and-structured-extraction-pipeline-stage-2)
  - [Phase 5 — Vagueness and specificity scoring (pipeline stage 3)](#phase-5--vagueness-and-specificity-scoring-pipeline-stage-3)
  - [Phase 6 — External evidence: registries and news (pipeline stage 4b)](#phase-6--external-evidence-registries-and-news-pipeline-stage-4b)
  - [Phase 7 — Verification (pipeline stage 5)](#phase-7--verification-pipeline-stage-5)
  - [Phase 8 — Hallucination controls](#phase-8--hallucination-controls)
  - [Phase 9 — Aggregation, audit report, and UI (pipeline stage 6)](#phase-9--aggregation-audit-report-and-ui-pipeline-stage-6)
  - [Phase 10 — Full evaluation, ablations, error analysis](#phase-10--full-evaluation-ablations-error-analysis)
- [Part D — Datasets](#part-d--datasets)
- [Part E — Evaluation framework](#part-e--evaluation-framework)
- [Part F — LangSmith and observability](#part-f--langsmith-and-observability)
- [Part G — Timeline, risks, and paper outline](#part-g--timeline-risks-and-paper-outline)

---

# Part A — Analysis of the existing project

## A.1 What `rag_app/` is today

A financial-document RAG system: upload a PDF, index it, ask questions, get
answers with page-level citations. The ingestion and retrieval half is solid and
directly reusable; the climate-specific half (claims, verification, external
evidence) does not exist yet.

| Component | File(s) | What it does |
|---|---|---|
| PDF extraction | [ingest.py](rag_app/src/ingest.py) | PyMuPDF text, tables, images; multi-row header merging; section propagation |
| Table fallback | [table_reconstruct.py](rag_app/src/table_reconstruct.py) | Rebuilds tables geometrically when `find_tables()` fails |
| Chunking | [chunker.py](rag_app/src/chunker.py) | OpenDataLoader typed elements; tables kept whole, header repeated into every split |
| Enrichment | [enrichment.py](rag_app/src/enrichment.py) | Regex-only; flattens rows into `row \| column: value` facts |
| Summaries | [summarizer.py](rag_app/src/summarizer.py) | Extractive page and document summaries (no LLM) |
| Retrieval | [retriever.py](rag_app/src/retriever.py), [hierarchical.py](rag_app/src/hierarchical.py) | BM25 + FAISS (exact cosine) fused with RRF; chunk / page / document tiers; multi-document |
| Reranking | [rerank.py](rag_app/src/rerank.py) | Hosted NVIDIA cross-encoder — **retired, defaults to off** |
| Generation | [llm.py](rag_app/src/llm.py) | NVIDIA NIM (llama-3.1-70b) or Gemini; temperature 0; strict grounding prompt |
| Query orchestration | [intelligence.py](rag_app/src/intelligence.py) | Routing, refusal floor, hedge detection, citation repair |
| API and UI | [server.py](rag_app/server.py), [static/index.html](rag_app/static/index.html), [app.py](rag_app/app.py) | FastAPI + HTML UI; Streamlit UI |
| Evaluation | [tests/eval.py](rag_app/tests/eval.py), [tests/golden_AEL.jsonl](rag_app/tests/golden_AEL.jsonl) | Single-PDF golden set; substring-match checks |

## A.2 Mapping to the six pipeline stages of the build plan

| Build-plan stage | Existing support | Verdict |
|---|---|---|
| 1. Ingestion and parsing | ingest, table_reconstruct, chunker, enrichment | **Reuse and extend** — strongest existing asset; emissions data lives in tables |
| 2. Claim detection and extraction | none | **New** |
| 3. Vagueness scoring | none | **New** |
| 4. Evidence retrieval | retriever, hierarchical | **Reuse** for in-report evidence; add registry and news stores, a working reranker, metadata filters |
| 5. Verification | grounded answering only | **New** verifier and numeric checker |
| 6. Aggregation and audit report | FastAPI UI, Streamlit app | **Reuse** UI shell; add claim → evidence → verdict views |
| Evaluation | substring checks on one PDF | **Replace** with a per-stage evaluation suite |

## A.3 Problems in the current code that affect research validity

These must be fixed before any number is reported (Phase 0).

| # | Problem | Location | Why it matters |
|---|---|---|---|
| 1 | Eval runs a different pipeline from the one served | [eval.py:32](rag_app/tests/eval.py#L32) uses `ConversationalRAG`; [server.py:46](rag_app/server.py#L46) serves `DocumentIntelligenceRAG` | Reported numbers would describe a pipeline nobody uses |
| 2 | Unmatched citations are kept, not dropped | [intelligence.py:692](rag_app/src/intelligence.py#L692) — `by_page_chunk.get(key, match.group(0))` | Invented citation ids pass through; corrupts citation-correctness metrics |
| 3 | No reranker | [rerank.py](rag_app/src/rerank.py) — hosted endpoint retired | The plan's "hybrid + cross-encoder" configuration cannot be run |
| 4 | All models are hosted APIs | [llm.py](rag_app/src/llm.py), [embeddings.py](rag_app/src/embeddings.py) | Results are not reproducible once a provider changes or retires a model (already happened with the reranker) |
| 5 | Chunks embedded without their section heading | `retriever.build()`; noted in [INGESTION_PIPELINE.md](rag_app/INGESTION_PIPELINE.md) | "Scope 1" in a table means little without "Principle 6 — Environment" |
| 6 | Refusal detected by a phrase list | [eval.py:120](rag_app/tests/eval.py#L120) | Not an acceptable metric for a paper |

## A.4 Design decisions carried forward

- **Stay in plain Python; do not migrate to LangChain or LlamaIndex.** The
  existing module boundaries make each ablation a function argument rather than a
  framework configuration, which is what a paper needs.
- **Keep** the table-aware chunking, `row | column: value` enrichment, RRF fusion,
  temperature 0, binary refusal and the `FORBIDDEN` hedge rules in
  [llm.py](rag_app/src/llm.py#L58). All of these carry over to climate reports.
- **Keep** `expand_fiscal_terms` in [chat.py](rag_app/src/chat.py) — Indian
  fiscal-year handling (FY 2022-23) is directly useful for BRSR.
- **All arithmetic runs in Python, never in the LLM.**

---

# Part B — Target system and research framing

## B.1 Core idea

Verify each corporate climate claim against outside evidence, not just its
wording. Every claim is checked against three evidence sources:

1. **The rest of the same report** — do reported emissions support the pledge?
2. **External registries** — is the target validated? What do independent estimates show?
3. **News and controversy sources** — accusations or litigation that contradict the claim?

Each claim gets one verdict, cited evidence, and a vagueness score; claims roll
up into a company-level risk score.

| Verdict | Meaning |
|---|---|
| Supported | Evidence backs the claim |
| Contradicted | Evidence conflicts with the claim |
| Unverifiable | No evidence found either way |
| Vague | Too unspecific to check (no number, scope or date) |

## B.2 End-to-end flow

```
 PDF (BRSR / sustainability report)
   │
 [1] Ingestion ──────────── text, tables, metadata (company, FY, section, page)
   │
 [2] Claim detection ────── ClimateBERT filter → LLM fills fixed JSON schema
   │
 [3] Vagueness scoring ──── ClimateBERT specificity + hedging lexicon → 0–1
   │
 [4] Evidence retrieval ─── (a) in-report hybrid index
   │                        (b) registry lookups (exact, not embedded)
   │                        (c) news hybrid index
   │                        supporting + refuting queries → cross-encoder rerank
   │
 [5] Verification ───────── Python numeric checker → LLM judge (structured JSON)
   │                        → hallucination guards (span, ID, NLI, consistency)
   │
 [6] Aggregation ────────── company risk score → audit report / UI
```

## B.3 Research questions

| RQ | Question | Primary data |
|---|---|---|
| RQ1 | Does hybrid retrieval with cross-encoder reranking beat BM25 alone and dense alone on corporate climate disclosures? | ClimRetrieve |
| RQ2 | How much does each evidence source (in-report, registry, news) improve verdict accuracy? | EmeraldData, BRSR set |
| RQ3 | Does deterministic numeric checking beat LLM arithmetic for target-progress claims? | BRSR set |
| RQ4 | Do layered hallucination controls improve faithfulness, and at what cost in coverage? | BRSR set, EmeraldData |

## B.4 Intended contributions

1. A **claim schema / taxonomy** for corporate climate claims (type, scope, baseline, target, conditionality, offset reliance).
2. The **first annotated greenwashing-verification dataset on Indian BRSR reports**, with numeric gold values from XBRL filings.
3. An **evidence-grounded verification pipeline** with deterministic numeric checking.
4. An **ablation of hallucination controls** with faithfulness and risk-coverage analysis.

---

# Part C — Build phases (logical sequence)

Each phase lists its goal, tasks, files, outputs and a **done when** criterion.
Phases are ordered so that every phase only depends on earlier ones, and so that
a benchmark result exists as early as possible (Phase 3).

Annotation (Phase 1) runs **in parallel** with Phases 2–8, because it is the
slowest human task.

```
Phase 0 Foundations ──► Phase 2 Ingestion ──► Phase 3 Retrieval + ClimRetrieve eval
        │                                          │
Phase 1 Data + annotation (parallel) ──────────────┤
                                                   ▼
                     Phase 4 Claims ──► Phase 5 Vagueness
                                                   │
                     Phase 6 Registry + news ◄─────┘
                                │
                     Phase 7 Verification ──► Phase 8 Hallucination controls
                                                   │
                     Phase 9 Aggregation + UI ◄────┘
                                │
                     Phase 10 Full evaluation, ablations, error analysis ──► Paper
```

---

## Phase 0 — Foundations and reproducibility

**Goal:** a reproducible, local-model codebase with an evaluation harness that
measures the pipeline actually being built.

### Tasks

1. **Fix the three correctness bugs** from [A.3](#a3-problems-in-the-current-code-that-affect-research-validity):
   - Point the eval harness at `DocumentIntelligenceRAG` (the served pipeline).
   - In `_repair_citations()` ([intelligence.py:680](rag_app/src/intelligence.py#L680)), **drop** citations that do not map to a retrieved chunk instead of keeping them.
   - Add `RERANK_BACKEND=local` in [rerank.py](rag_app/src/rerank.py) using `BAAI/bge-reranker-v2-m3` via `sentence-transformers` `CrossEncoder`.
2. **Move to local, versioned models** (keep the API backends as an optional comparison):

   | Role | Current | Target |
   |---|---|---|
   | Generator / judge | NIM llama-3.1-70b (API) | Open 7–8B instruct model via Ollama or vLLM (e.g. Qwen3-8B or Llama-3.1-8B) |
   | Comparison model | — | One API model (Gemini / GPT / Claude) |
   | Embeddings | nv-embedqa-e5-v5 (API) | `BAAI/bge-m3` or `intfloat/e5-large-v2` (local) |
   | Reranker | none | `BAAI/bge-reranker-v2-m3` (local) |
   | NLI checker | none | DeBERTa-v3 NLI or Vectara HHEM-2.1 (local) |

   Add an `ollama` / `vllm` backend to `llm.py` through the existing
   OpenAI-compatible client path (`_nvidia_client()` already uses the `openai`
   SDK with a configurable `base_url`).
3. **Pin everything:** model names and revisions, `requirements.txt` versions, random seeds. Record them in every run's output file.
4. **Create the evaluation skeleton:** an `eval/` folder where each script reads a dataset, runs one stage, and writes `results/<experiment>/<timestamp>.json` with metrics, config and model versions.
5. **Add tracing** (LangSmith or Phoenix — see [Part F](#part-f--langsmith-and-observability)).
6. **Create the new package layout:**

```
rag_app/src/
  ingest.py, chunker.py, enrichment.py         # Stage 1 (modify)
  claims/schema.py, detect.py, extract.py      # Stage 2 (new)
  claims/vagueness.py                          # Stage 3 (new)
  evidence/report_index.py                     # Stage 4a (wraps retriever/hierarchical)
  evidence/registry.py, news_index.py          # Stage 4b (new)
  evidence/queries.py                          # supporting + refuting query builder
  rerank.py                                    # add local cross-encoder
  verify/numeric_check.py, judge.py, guards.py # Stage 5 + hallucination controls
  audit/aggregate.py, report.py                # Stage 6
eval/
  retrieval_climretrieve.py   claims_envclaims.py   claims_netzero.py
  extraction_xbrl.py          vagueness_eval.py     verdicts_emerald.py
  verdicts_brsr.py            faithfulness.py       ablations.py
  stats.py                    # bootstrap CIs, McNemar
data/
  raw/ processed/ registry/ news/
  annotation/guidelines.md  brsr_claims_v1.jsonl
results/
```

### Done when

- The existing golden set runs through `DocumentIntelligenceRAG` on local models end to end.
- The reranker status reports `active` with the local backend.
- A run writes a results JSON that includes the model versions.

---

## Phase 1 — Data collection and annotation setup

**Goal:** all datasets downloaded and versioned; BRSR corpus collected;
annotation guideline drafted and piloted. Runs in parallel from week 1.

### Tasks

1. **Download component benchmarks** (see [Part D](#part-d--datasets)) into `data/raw/` and record their versions or commit hashes.
2. **Build the BRSR corpus:**
   - 30–50 NIFTY companies, spread across sectors with high emissions (cement, steel, power, oil & gas, automobiles) and low emissions (IT, banking, FMCG).
   - FY 2022-23, 2023-24 and 2024-25 BRSR PDFs (from NSE/BSE filings or company sites).
   - **The matching BRSR XBRL filings** — these give machine-readable gold values for Scope 1/2 emissions, energy and water. Confirm XBRL coverage for each year and company.
   - A `companies.csv` with name, NSE symbol, ISIN, CIN, sector. **This is the join key across all sources.**
3. **Collect registry snapshots** (dated, stored in `data/registry/`): SBTi Target Dashboard export, Net Zero Tracker export, Climate TRACE data for Indian assets.
4. **Collect news:** GDELT queries per company (name + aliases) plus a curated list of known controversies; store with publication date, URL and source.
5. **Write the annotation guideline** (`data/annotation/guidelines.md`):
   - Definition of a climate claim; the claim schema fields; the four verdicts with decision rules and worked examples; what counts as evidence; the time window for evidence.
6. **Pilot annotation:** two annotators label the same ~50 claims, compute Cohen's κ, settle disagreements, revise the guideline. Repeat until κ ≥ 0.6.

### Done when

- All datasets are in `data/raw/` with a `DATASETS.md` that lists source, version, licence and size.
- BRSR PDFs and XBRL are collected for at least 30 companies × 2 years.
- The guideline is finalised after a pilot with κ ≥ 0.6.

---

## Phase 2 — Ingestion upgrade (pipeline stage 1)

**Goal:** BRSR and sustainability reports parse into chunks with reliable
company / year / section metadata and correct emissions tables.

### Tasks

1. **Metadata on every chunk:** add `company_id` (ISIN), `fiscal_year`, `report_type` (brsr / sustainability / annual), and `source_type` (report / registry / news) to `Chunk` ([chunker.py](rag_app/src/chunker.py)) and `DocMeta` ([service.py](rag_app/src/service.py#L74)).
2. **BRSR-aware sectioning:** extend `_guess_section()` in [ingest.py](rag_app/src/ingest.py) to recognise BRSR's fixed structure — Section A / B / C and Principles 1–9. **Principle 6 (Environment)** contains energy, Scope 1/2 emissions, water and waste tables.
3. **Section-prefixed embeddings:** in `retriever.build()`, embed `f"{section}\n{text}"` instead of `text` alone. The displayed chunk text stays unchanged.
4. **Climate enrichment** in [enrichment.py](rag_app/src/enrichment.py):
   - Unit regexes: `tCO2e`, `tCO2`, `GJ`, `TJ`, `MWh`, `kL`, `%`, intensity per ₹ crore of turnover.
   - Scope labels (Scope 1 / 2 / 3), baseline and target years.
   - Normalise units into a canonical form (e.g. all emissions → tCO2e) with the conversion recorded.
5. **Structured emissions table extraction:** a function that turns Principle 6 tables into rows `{company, fy, metric, scope, value, unit, page}`. These rows feed both the numeric checker (Phase 7) and the XBRL extraction benchmark.
6. **Bump `CHUNKER_VERSION`** so all indexes rebuild.
7. **Optional:** compare PyMuPDF / OpenDataLoader with Docling on 10 reports for table accuracy; keep the better one as the default.

### Evaluation in this phase

| What | Data | Metric |
|---|---|---|
| Emissions number extraction | BRSR XBRL gold values | Exact-match accuracy; accuracy within ±1% tolerance; per-field (scope, unit, year) accuracy |
| (Optional) cross-check | GHG emission extraction gold set (139 reports) | Same metrics |

### Done when

- 30+ BRSR reports are indexed with section metadata.
- Extracted Scope 1/2 values match XBRL at a measured, reported accuracy.

---

## Phase 3 — In-report retrieval and the first benchmark (pipeline stage 4a)

**Goal:** a measured, tuned in-report retriever and the first paper result (RQ1).

### Tasks

1. Wrap `HierarchicalRetriever` / `MultiDocumentRetriever` in `evidence/report_index.py` with **metadata filters** (`company_id`, `fiscal_year`).
2. Implement four configurations behind one flag:
   `bm25` · `dense` · `hybrid` (RRF) · `hybrid+rerank`.
3. Build the ClimRetrieve evaluation: index its 30 reports, run its 16 questions, score against its relevance labels.
4. Tune `candidate_k`, `rrf_k`, and rerank depth **on a dev split only**.

### Evaluation in this phase

| What | Data | Metric | Tool |
|---|---|---|---|
| Retrieval | ClimRetrieve | Recall@5/10, nDCG@10, MRR | `ranx` or `pytrec_eval` |
| Section-prefix effect | ClimRetrieve | Same metrics, with vs. without prefix | — |

### Done when

- A results table compares the four configurations, with bootstrap confidence intervals.
- This table is the RQ1 result in the paper.

---

## Phase 4 — Claim detection and structured extraction (pipeline stage 2)

**Goal:** from a report, produce a list of structured, verifiable claims with a
link back to their source text.

### Tasks

1. **Detection** (`claims/detect.py`): run `climatebert/environmental-claims` locally over every sentence of text chunks; keep sentences above a threshold tuned on the dev split. This cheap filter keeps most sentences away from the LLM.
2. **Schema** (`claims/schema.py`) — a Pydantic model; describe it carefully in the paper:

   ```python
   class Claim(BaseModel):
       claim_id: str
       company_id: str
       fiscal_year: str
       page: int
       source_chunk_id: int
       verbatim_span: str            # must be a substring of the source chunk
       claim_type: Literal["net_zero", "pct_reduction", "renewable_share",
                           "offset_based", "intensity", "generic"]
       scope: list[Literal["1", "2", "3"]] | None
       baseline_year: int | None
       target_year: int | None
       metric: str | None            # e.g. "absolute Scope 1+2 emissions"
       value: float | None           # e.g. 40.0
       unit: str | None              # e.g. "%", "tCO2e"
       is_quantified: bool
       is_conditional: bool          # "subject to…", "if policy allows…"
       relies_on_offsets: bool
   ```

3. **Extraction** (`claims/extract.py`): the LLM fills the schema with **constrained JSON output** (Ollama `format=<json schema>` or vLLM guided decoding). Validate with Pydantic; retry once on failure, then discard.
4. **Span check:** reject any claim whose `verbatim_span` is not a substring of the source chunk (after normalising whitespace).
5. **Deduplicate** claims repeated across pages (the same target often appears in the chairman's letter and in Principle 6).
6. **Hand-check ~100 extracted claims** for field-level accuracy.

### Evaluation in this phase

| What | Data | Metric |
|---|---|---|
| Claim detection | `climatebert/environmental_claims` | Precision, recall, F1 (compare with the published baseline) |
| Claim type | `climatebert/netzero_reduction_data` | Macro-F1 on net-zero vs. reduction target |
| Schema fields | ~100 hand-checked BRSR claims | Exact match per field; numeric value accuracy |
| Span validity | All extracted claims | % passing the substring check (reported as an extractor quality measure) |

### Done when

- Every BRSR report yields a claims JSONL file.
- Detection F1 and field-level accuracy are reported.

---

## Phase 5 — Vagueness and specificity scoring (pipeline stage 3)

**Goal:** a calibrated 0–1 vagueness score per claim, and the *Vague* verdict.

### Tasks

1. Features per claim:
   - ClimateBERT `climate_specificity` probability
   - ClimateBERT `climate_commitments_actions` probability (commitment vs. action taken)
   - Hedging lexicon hits: "aim to", "strive", "endeavour", "where feasible", "aspire", "committed to exploring", "subject to" …
   - Schema completeness: missing scope, baseline year, target year, value
2. Combine with **logistic regression trained on a labeled dev set**, not hand-picked weights, so the score is calibrated and defensible.
3. Pick a threshold for assigning *Vague*, chosen on dev.
4. Report a **reliability diagram** (calibration) for the score.

### Evaluation in this phase

| What | Data | Metric |
|---|---|---|
| Specificity component | `climatebert/climate_specificity` | F1 |
| Vague verdict | BRSR annotated set (dev / test) | Precision, recall, F1 for the *Vague* class |
| Calibration | BRSR dev set | Expected calibration error, reliability diagram |

### Done when

- Every claim has `vagueness_score` and a `vague` flag.

---

## Phase 6 — External evidence: registries and news (pipeline stage 4b)

**Goal:** for each claim, gather evidence from all three sources, including
evidence that could contradict it.

### Tasks

1. **Registry store** (`evidence/registry.py`) — **exact lookups, not embeddings.** Registry data is already structured, so embedding table rows only adds retrieval error.
   - Load SBTi, Net Zero Tracker and Climate TRACE snapshots into pandas / SQLite keyed by ISIN.
   - Functions such as `sbti_status(company_id)`, `nzt_pledge(company_id)`, `climate_trace_emissions(company_id, year)`.
   - Each result becomes an evidence item with a stable ID, source name and snapshot date.
2. **News index** (`evidence/news_index.py`) — a separate hybrid index (reuse `HybridRetriever`) over news articles with metadata `company_id`, `published_at`, `source`; filter by company and a **defined time window** relative to the report date.
3. **Query builder** (`evidence/queries.py`) — build queries from the structured claim, in two directions:
   - **Supporting:** `"<company> Scope 1 emissions <baseline_year> baseline"`, `"<company> renewable energy share <fy>"`
   - **Refuting:** `"<company> emissions increase"`, `"<company> greenwashing"`, `"<company> environmental violation / litigation"`
4. **Leakage rule:** exclude the claim's own source chunk from its in-report evidence, so a claim can never be retrieved as support for itself.
5. **Rerank** all non-registry evidence with the local cross-encoder; keep top-k per source.
6. **Evidence record format** (shared by all sources):

   ```json
   {"evidence_id": "news:000123", "source_type": "news", "company_id": "INE...",
    "text": "...", "url_or_citation": "...", "date": "2024-06-02", "score": 0.81}
   ```

### Matching companies across sources

Company names differ across BRSR, SBTi, Net Zero Tracker, Climate TRACE and news
("Tata Steel Ltd" vs. "Tata Steel Limited" vs. a subsidiary). Use ISIN as the key,
keep an alias table, and hand-check every match for the 30–50 companies. Describe
this briefly in the paper.

### Done when

- For every claim, an evidence bundle with items from all three sources (or an explicit "none found") is written to disk.

---

## Phase 7 — Verification (pipeline stage 5)

**Goal:** one verdict per claim, with a rationale and cited evidence IDs.

### Tasks

1. **Numeric checker** (`verify/numeric_check.py`) — deterministic Python, no LLM:
   - Input: the claim (e.g. "−40% Scope 1+2 by 2030 vs. FY2020") plus the structured emissions rows from Phase 2 (all available years).
   - Compute the actual change from baseline to the latest year, the required annual rate, the observed annual rate, and a linear projection to the target year.
   - Output: `on_track` / `off_track` / `insufficient_data`, plus every intermediate number, so the result can be shown and audited.
   - **Cross-year checks:** flag a changed baseline year, a changed baseline value, or a moved target year between FY reports for the same company.
2. **LLM judge** (`verify/judge.py`):
   - Input: the claim, the evidence bundle (each item with its ID), and the numeric checker's output given **as a fact**.
   - Output (constrained JSON):

     ```json
     {"verdict": "Supported|Contradicted|Unverifiable|Vague",
      "rationale": ["sentence 1", "sentence 2"],
      "evidence_ids": ["report:p42:c118", "registry:sbti:..."],
      "quotes": [{"evidence_id": "...", "text": "exact substring"}],
      "confidence": 0.0}
     ```
   - Prompt rules carried over from [llm.py](rag_app/src/llm.py): evidence-only, no hedging, binary refusal → here, "no evidence" maps to *Unverifiable*.
3. Claims already flagged *Vague* in Phase 5 skip the judge (or are judged and reported separately — decide once, document it).

### Done when

- Every claim has a verdict record.
- The numeric checker has unit tests on hand-built trajectories (on track, off track, missing years, changed baseline).

---

## Phase 8 — Hallucination controls

**Goal:** make every verdict traceable to evidence, and turn unsupported outputs
into *Unverifiable* instead of confident errors. Each control is a separate,
switchable function in `verify/guards.py`, so each can be ablated.

| # | Control | Where | What it catches |
|---|---|---|---|
| 1 | **Constrained JSON output** (schema-guided decoding) | extract, judge | Malformed or free-form output |
| 2 | **Verbatim span check:** every quote must be a substring of its cited evidence | extract, judge | Invented quotes and claims |
| 3 | **Evidence ID whitelist:** drop IDs not in the retrieved bundle; if none remain → *Unverifiable* | judge | Invented citations |
| 4 | **All arithmetic in Python** | numeric_check | Wrong calculations |
| 5 | **NLI entailment check:** each rationale sentence must be entailed by its cited evidence (DeBERTa-v3 NLI or HHEM-2.1); failure → *Unverifiable* | after judge | Unsupported reasoning |
| 6 | **Self-consistency:** run the judge 3–5 times with the evidence order shuffled; disagreement → *Unverifiable* | judge | Unstable verdicts, position bias |
| 7 | **Counter-evidence retrieval** (supporting + refuting queries) | Phase 6 | Confirmation bias, over-trusting the company |
| 8 | **Exclude the claim's own chunk** from its evidence | Phase 6 | Circular support |
| 9 | **Company and year metadata filters** | retrieval | Mixing up companies or years |
| 10 | **Abstention threshold** on `confidence`, tuned on dev for a target precision | final | Overconfident guesses |
| 11 | **News treated as untrusted input** — wrapped in delimiters; the model is told not to follow instructions inside it | judge prompt | Prompt injection from scraped pages |

Existing controls to keep: temperature 0, the binary refusal rule and `FORBIDDEN`
hedge patterns in [llm.py](rag_app/src/llm.py#L58), the hedge detector
`_HALLUCINATION_RE` in [intelligence.py](rag_app/src/intelligence.py#L669), and the
fixed citation repair from Phase 0.

### Done when

- Each control can be switched on and off by config.
- One run with all controls on and one with all off have been compared on the BRSR dev set.

---

## Phase 9 — Aggregation, audit report, and UI (pipeline stage 6)

**Goal:** a company-level risk score and a demo that shows each claim, its
evidence and its verdict.

### Tasks

1. **Company risk score** (`audit/aggregate.py`):

   ```
   risk = w1 · mean(vagueness_score)
        + w2 · share(Contradicted)
        + w3 · share(Unverifiable)
        + w4 · cross_year_flags          # moved targets / changed baselines
   ```

   - Report a **sensitivity analysis** for the weights (e.g. Kendall's τ of the company ranking under changes to the weights). Reviewers will ask about this.
2. **Audit report** (`audit/report.py`): per company, a JSON + Markdown/HTML report: claims → evidence (with page links / URLs) → verdict → rationale → confidence.
3. **UI:** extend the existing FastAPI + [static/index.html](rag_app/static/index.html) (or the Streamlit [app.py](rag_app/app.py)) with:
   - a company list with risk scores,
   - a claims table with verdict and vagueness,
   - a claim view showing evidence and the PDF page (page serving already exists).
4. **Wording:** label outputs as **"risk flags for human review"**, never as accusations of greenwashing.

### Done when

- A full run on the BRSR corpus produces a report for every company, viewable in the UI.

---

## Phase 10 — Full evaluation, ablations, error analysis

**Goal:** the Results, Ablations and Error Analysis sections of the paper.

### Tasks

1. Freeze the test splits (they must not have been used for any tuning).
2. Run all stage-level evaluations from [Part E](#part-e--evaluation-framework) on test.
3. Run baselines and ablations (below).
4. Error analysis: take 20–30 failures and label each by cause — table parsing, retrieval miss, wrong company match, numeric extraction, judge over-trusting company wording, NLI false rejection, annotation ambiguity.
5. Produce all tables and figures from scripts in `eval/`, so they can be regenerated.

### Baselines

1. ClimateBERT classifier only (no LLM, no retrieval)
2. LLM without retrieval (zero-shot)
3. Standard RAG (the current pipeline, unchanged)
4. EmeraldMind (on EmeraldData)
5. Full pipeline

Note on baseline 2: the LLM may already know about a company's controversies from
its training data. Report this as a confound.

### Ablations

| Ablation | Answers |
|---|---|
| No retrieval → in-report only → + registry → + news | RQ2 |
| LLM arithmetic vs. Python numeric checker | RQ3 |
| Each hallucination control removed one at a time | RQ4 |
| BM25 / dense / hybrid / hybrid + rerank | RQ1 |
| Small open model vs. API model | Cost vs. quality |
| With vs. without section-prefixed embeddings | Ingestion design |

### Done when

- Every table and figure in the paper is produced by a script in `eval/`, from a committed results file.

---

# Part D — Datasets

## D.1 Component benchmarks

| Stage | Dataset | Size / content | Use |
|---|---|---|---|
| Claim detection | `climatebert/environmental_claims` (Stammbach et al.) | ~2.6K sentences, claim / not claim | Detector F1 |
| Claim type | `climatebert/netzero_reduction_data` (Schimanski et al. 2023) | Net-zero vs. reduction target labels | `claim_type` field evaluation (**not in the original build plan; added**) |
| Vagueness features | ClimateBERT `climate_specificity`, `climate_commitments_actions`, `climate_detection`, `climate_sentiment` | Sentence-level labels from corporate reports | Stage 3 features |
| Retrieval | ClimRetrieve (EMNLP 2024; GitHub `tobischimanski/ClimRetrieve`) | 30 reports, 16 questions, 8.5K+ relevance-labeled pairs | RQ1 |
| Number extraction | GHG emission extraction gold set (PMC12391291) | Emission metrics from 139 reports | Optional cross-check for Phase 2 |
| Talk vs. action | A3CG (NUS) | Statements by aspect and action type | Optional, for the commitments-vs-actions feature |
| Verifier sanity check | CLIMATE-FEVER | ~1.5K claims with evidence, supports / refutes | General-claim performance vs. corporate claims |

## D.2 End-to-end greenwashing evaluation

| Dataset | Content | Caveat |
|---|---|---|
| EmeraldData (`geoka/EmeraldData`) | 620 claims (225 greenwashing, 395 not) tied to ESG reports and news; from the EmeraldMind paper — closest related work and direct baseline | An LLM was the primary annotator → also build the human-labeled BRSR set |
| Climate policy engagement (NeurIPS 2023 D&B) | Evidence of corporate climate policy engagement from report PDFs | Optional; for the gap between what companies say and how they lobby |

## D.3 Ground-truth sources for verification

| Source | Use for Indian (BRSR) companies? | Use |
|---|---|---|
| SBTi Target Dashboard | Yes | Is the target validated? |
| Net Zero Tracker | Yes | Pledge structure and quality |
| Climate TRACE | Yes (covers Indian facilities) | Independent emissions estimates |
| BRSR Core assurance | Yes | Third-party assured values |
| BRSR XBRL filings | Yes | Gold values for numeric extraction |
| US EPA GHGRP | **No** — US facilities only | Only for global companies in EmeraldData |
| EU ETS | **No** — EU installations only | Only for global companies in EmeraldData |
| GDELT / curated news | Yes | Accusations, controversies, litigation |

## D.4 Our contribution: the Indian BRSR dataset

- **Why:** SEBI has required BRSR from India's top listed companies since FY 2022-23, and BRSR Core carries third-party assurance. Almost all greenwashing NLP work uses US/EU reports.
- **Corpus:** 30–50 companies × 2–3 fiscal years.
- **Annotation:** ~300 claims, two annotators, written guideline, pilot round, Cohen's κ reported, disagreements settled by discussion.
- **Splits:** by **company**, not by claim (dev ~30%, test ~70%), so no company appears in both — otherwise the model is tested on companies it was tuned on.
- **Bonus findings:** cross-year changes (moved targets, changed baselines), and XBRL-based numeric extraction accuracy on Indian reports.
- **Release:** publish the claim annotations and scripts; link to the source PDFs rather than redistributing them if licensing is unclear.

---

# Part E — Evaluation framework

## E.1 Metrics per stage

| Stage | Metric | Tool |
|---|---|---|
| Retrieval | Recall@5/10, nDCG@10, MRR | `ranx` / `pytrec_eval` |
| Claim detection | Precision, recall, F1 | scikit-learn |
| Schema extraction | Exact match per field; numeric accuracy within ±1% | own script |
| Table numbers | Accuracy against XBRL gold | own script |
| Vagueness | F1 for *Vague*; expected calibration error | scikit-learn |
| Verdicts | **Macro-F1** (imbalanced classes), per-class F1, confusion matrix | scikit-learn |
| Faithfulness | RAGAS faithfulness and context precision; **NLI entailment rate** of rationale sentences | ragas, HHEM / AlignScore |
| Citations | **Citation precision** (cited evidence supports the sentence) and **citation recall** (every sentence is supported by some citation), ALCE-style | NLI model + ~100 human-checked samples |
| Abstention | **Risk-coverage curve**: accuracy vs. share of claims given a non-*Unverifiable* verdict | own script |
| Annotation quality | Cohen's κ (2 annotators); Fleiss' κ if more | scikit-learn / statsmodels |
| Company risk score | Kendall's τ under changes to the weights | scipy |

## E.2 Statistical rigour

- **Bootstrap 95% confidence intervals** (1,000+ resamples) on every headline number.
- **McNemar's test** or a paired bootstrap for any "system A beats system B" claim on the same items.
- **3 runs** for anything stochastic (self-consistency sampling, any non-zero temperature); report mean ± std.
- **Validate any LLM-as-judge metric** (including RAGAS) against ~100 human labels and report agreement before relying on it.
- **Tune on dev only.** Test splits are evaluated once per final configuration.

## E.3 What every results file must record

```json
{"experiment": "verdicts_brsr_full", "split": "test", "timestamp": "...",
 "git_commit": "...", "models": {"judge": "...", "embed": "...", "rerank": "...", "nli": "..."},
 "config": {"top_k": 8, "guards": ["span", "id", "nli", "consistency"]},
 "metrics": {"macro_f1": 0.0, "macro_f1_ci": [0.0, 0.0]}, "n": 0}
```

---

# Part F — LangSmith and observability

## F.1 Can we use LangSmith? Yes — with a clear division of roles

LangSmith works **without LangChain**:

- Wrap the OpenAI-compatible client with `langsmith.wrappers.wrap_openai()` (works for Ollama, vLLM and NIM, which all expose OpenAI-compatible endpoints; the existing `_nvidia_client()` already uses the `openai` SDK).
- Decorate pipeline functions with `@traceable`: `detect`, `extract`, `retrieve`, `rerank`, `numeric_check`, `judge`, each guard.

## F.2 Use it for

- **Tracing** every claim through all six stages — makes error analysis fast ("why did claim 47 come out *Contradicted*?").
- **Datasets and experiments:** upload dev sets and compare pipeline variants side by side.
- **Custom evaluators:** run our own Python metrics (macro-F1, citation precision) over each experiment.

## F.3 Do not use it for

- **The paper's numbers.** These come from versioned scripts in `eval/` that write results files to the repo. Reviewers must be able to reproduce results without our LangSmith account.
- Built-in LLM-as-judge scores, unless validated against human labels.

## F.4 Caveats and alternatives

| Tool | Hosting | Notes |
|---|---|---|
| LangSmith | Hosted SaaS | Easiest experiment comparison; data leaves the machine; free tier has trace limits (check current quota) |
| Arize Phoenix | Local, open source | Fully local tracing and evals; integrates with RAGAS |
| Langfuse | Self-hostable, open source | Tracing, datasets, evals |

**Recommendation:** LangSmith (or Phoenix if data must stay local) for development
and debugging; plain scripts plus committed results files for the paper. Keep the
existing Prometheus metrics in [observability.py](rag_app/src/observability.py) for
the running service.

---

# Part G — Timeline, risks, and paper outline

## G.1 Ten-week timeline

| Week | Build | Data / annotation (parallel) |
|---|---|---|
| 1 | Phase 0: bug fixes, local models, eval skeleton, tracing | Phase 1: download benchmarks; start BRSR + XBRL collection |
| 2 | Phase 2: metadata, BRSR sectioning, section-prefixed embeddings | Company alias table; registry snapshots; draft guideline |
| 3 | Phase 2: climate enrichment, emissions table extraction, XBRL eval | Pilot annotation (50 claims), κ, guideline revision |
| 4 | Phase 3: retrieval configs + ClimRetrieve eval (**RQ1 result**) | News collection |
| 5 | Phase 4: claim detection + extraction + evals | Annotation round 1 |
| 6 | Phase 5: vagueness; Phase 6: registry + news + query builder | Annotation round 1 |
| 7 | Phase 7: numeric checker + judge; Phase 8: guards | Annotation round 2, κ |
| 8 | Phase 9: aggregation, report, UI | Adjudication; freeze test split |
| 9 | Phase 10: full evaluation, baselines, ablations, error analysis | — |
| 10 | Paper writing; release dataset and code | — |

## G.2 Risks and mitigations

| Risk | Mitigation |
|---|---|
| BRSR tables parse badly | XBRL gold values expose errors early (Phase 2); Docling comparison; `table_reconstruct.py` fallback |
| Annotation is slow or inconsistent | Start in week 1; pilot round; written guideline; split by company |
| Registry coverage of Indian companies is thin | Many claims will be *Unverifiable* from registries alone — report coverage per source honestly; in-report and news evidence still apply |
| Company name matching errors | ISIN key + alias table + manual check of all 30–50 companies |
| Hosted models change mid-project | Local pinned models are the primary results; API model only as a comparison |
| LLM training-data leakage | No-retrieval baseline measures it; discuss in Limitations |
| Harm from false greenwashing accusations | Frame outputs as risk flags for human auditors; report precision of *Contradicted* specifically; Ethics section |

## G.3 Paper outline

1. **Introduction** — claim verification against evidence, not just claim wording
2. **Related Work** — climate NLP (ClimateBERT, ChatReport, ClimRetrieve), greenwashing detection (EmeraldMind, A3CG, environmental claims), RAG fact-checking and attribution (CLIMATE-FEVER, ALCE-style citation evaluation)
3. **Problem Formulation and Claim Taxonomy** — the schema (contribution)
4. **Dataset** — the BRSR corpus, XBRL gold values, annotation protocol, κ (contribution)
5. **Method** — the six-stage pipeline; numeric checker; hallucination controls
6. **Experiments** — datasets, baselines, metrics, statistical tests
7. **Results and Ablations** — RQ1–RQ4
8. **Error Analysis** — categorised failures
9. **Limitations and Ethics** — outputs are risk flags, not verdicts; registry coverage; LLM-labeled external data; training-data leakage
10. **Conclusion**

**Target venues:** ClimateNLP workshop at ACL; NeurIPS Tackling Climate Change with ML workshop.

## G.4 First actions

1. Fix the three Phase 0 bugs (eval pipeline, citation dropping, local reranker).
2. Add the local model backends.
3. Start BRSR + XBRL collection and the annotation guideline the same week.
