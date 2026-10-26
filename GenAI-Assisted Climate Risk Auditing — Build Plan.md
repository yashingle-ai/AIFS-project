# GenAI-Assisted Climate Risk Auditing — Build Plan

Oct 25, 2026 · @Tarun

## Core idea

The system verifies each corporate climate claim against outside evidence, not just the claim's wording. Most student work stops at classifying sentences as vague; checking claims against evidence is what makes this a research contribution.

Every claim is checked against three evidence sources:

1. **The rest of the same report.** Do the reported emissions numbers actually support the pledge?
2. **External registries.** Is the target validated? What do independent emissions estimates show?
3. **News and controversy sources.** Are there accusations or litigation that contradict the claim?

Each claim then gets one verdict, cited evidence, and a risk score.

| Verdict | Meaning |
| --- | --- |
| Supported | Evidence backs the claim |
| Contradicted | Evidence conflicts with the claim |
| Unverifiable | No evidence found either way |
| Vague | Too unspecific to check (no number, scope or date) |

## Pipeline architecture

The pipeline has six stages, from PDF to audit report. The design rules that matter most: a cheap classifier filters sentences before the LLM sees them, and all arithmetic runs in Python, not in the LLM.

&#91;embedded content: pipeline architecture · 6 stages, 3 evidence indexes\]

Stages 1–3 turn a report into structured claims. Stage 4 pulls evidence from three indexes, stage 5 checks each claim, and stage 6 scores the company.

1. **Ingestion and parsing**
   - Parse PDFs with Docling or PyMuPDF. Use Camelot or pdfplumber for tables, since emissions data lives in tables.
   - Chunk by section heading, not fixed size.
   - Keep metadata on each chunk: company, year, page, section, and text vs. table.
2. **Claim detection and structured extraction**
   - Filter candidate sentences with a fine-tuned ClimateBERT environmental-claims classifier (cheap and fast).
   - Send only flagged sentences to an LLM, which fills a fixed JSON schema:
     - claim type: net-zero, % reduction, renewable share, offset-based, or generic
     - scope (1/2/3), baseline year, target year, metric, value
     - flags: quantified? conditional ("subject to…")? relies on offsets?
   - The schema is a contribution in itself, so describe it carefully in the paper.
3. **Vagueness and specificity scoring**
   - Use the ClimateBERT specificity and commitments-vs-actions models.
   - Add a hedging-language lexicon ("aim to", "strive", "where feasible").
   - Output a 0–1 vagueness score per claim.
4. **Evidence retrieval (the RAG core)**
   - Hybrid retrieval: BM25 plus a dense embedding model (bge or e5 family), then a cross-encoder reranker.
   - Separate indexes for in-report evidence, external registry and emissions data, and news.
   - Build queries from the structured claim, e.g. "Scope 1 emissions 2019 baseline Company X".
5. **Verification**
   - An LLM judge receives the claim plus top-k evidence. It returns a verdict, a rationale, and the IDs of the evidence it used.
   - Numeric checks run in Python. Example: does the reported Scope 1 trajectory put the company on track for "-40% by 2030"? A deterministic checker is also a clean ablation for the paper.
6. **Aggregation and audit report**
   - Combine per-claim verdicts and vagueness scores into a company-level greenwashing risk score.
   - A Streamlit dashboard shows each claim → evidence → verdict, with page citations.

**Tech stack**

| Layer | Tools |
| --- | --- |
| Language and orchestration | Python, LangChain or LlamaIndex |
| Vector store | FAISS or Qdrant |
| Embeddings and classifiers | sentence-transformers, HuggingFace ClimateBERT models |
| LLMs | One open 7–8B model via Ollama or vLLM; optionally one API model for comparison |
| Evaluation | RAGAS (faithfulness, context precision) |
| Demo | Streamlit |

## Datasets

Use public benchmarks to evaluate each stage, EmeraldData for end-to-end greenwashing detection, and your own Indian BRSR corpus as the novel contribution.

### Component benchmarks

| Dataset | What it gives you | Use it for |
| --- | --- | --- |
| Environmental Claims (Stammbach et al.; HF `climatebert/environmental_claims`) | About 2.6K sentences labeled claim / not-claim | Stage 2 claim detector |
| ClimateBERT datasets (`climate_detection`, `climate_specificity`, `climate_commitments_actions`, `climate_sentiment`) | Sentence-level labels from corporate reports | Stage 3 vagueness features |
| CLIMATE-FEVER | About 1.5K climate claims with evidence and supports/refutes labels | Sanity-checking the verifier |
| [ClimRetrieve](https://arxiv.org/pdf/2406.09818) (EMNLP 2024) | 30 sustainability reports, 16 climate questions, 8.5K+ question-source-answer pairs labeled by relevance; code on GitHub (tobischimanski/ClimRetrieve) | Best benchmark for the retrieval stage |
| [GHG emission extraction gold set](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12391291/) | Emission metrics from 139 sustainability reports | Table and number extraction |
| [A3CG](https://cde.nus.edu.sg/ai-turns-the-tide-on-greenwashing) (NUS) | Sustainability statements annotated by aspect and the type of action taken | Separating talk from action |

### End-to-end greenwashing evaluation

- [EmeraldData](https://huggingface.co/datasets/geoka/EmeraldData/blob/main/README.md) (HF `geoka/EmeraldData`): 620 claims, 225 labeled greenwashing and 395 not, anchored to real ESG reports and news. It comes from the [EmeraldMind](https://arxiv.org/pdf/2512.11506v2) paper, a RAG framework for greenwashing detection, so it is your closest related work and a direct baseline. Caveat for the paper: an LLM was the primary annotator, which is a reason to also build a small human-labeled test set.
- [Climate policy engagement dataset](https://papers.neurips.cc/paper_files/paper/2023/file/7ccaa4f9a89cce6619093226f26b84e6-Paper-Datasets_and_Benchmarks.pdf) (NeurIPS 2023 Datasets & Benchmarks, climate-nlp.github.io): extracts evidence of corporate climate policy engagement from report PDFs. Useful for the gap between what companies say and how they lobby.

### Ground-truth sources for verification

| Source | What it tells you |
| --- | --- |
| SBTi Target Dashboard | Whether a company's target is validated by the Science Based Targets initiative |
| Net Zero Tracker | Structured data on company net-zero pledges and their quality |
| Climate TRACE | Independent global emissions estimates |
| US EPA GHGRP | Reported facility-level emissions in the US |
| EU ETS | Verified installation-level emissions in the EU |
| GDELT or curated news | Accusations, controversies and litigation per company |

### Your differentiator: Indian BRSR reports

Since 2022–23, SEBI has required India's top listed companies to file Business Responsibility and Sustainability Reports (BRSR); the BRSR Core subset carries third-party assurance. Almost all greenwashing NLP work uses US/EU reports.

- Build a corpus of 30–50 Indian companies across 2–3 years.
- Annotate about 300 claims yourselves.
- Use cross-year comparison to catch quietly moved targets and changed baselines.

## Evaluation plan

Each pipeline stage gets its own metric, and the ablations become the Results section of the paper.

| What is evaluated | Data | Metric |
| --- | --- | --- |
| Retrieval | ClimRetrieve | Recall@k, nDCG; compare BM25 vs. dense vs. hybrid vs. hybrid + reranker |
| Claim extraction | Environmental Claims; about 100 hand-checked claims | F1; field-level accuracy of the JSON schema |
| Verification | EmeraldData; your own annotated set | Macro-F1; Cohen's κ between two annotators |
| Faithfulness | Pipeline outputs | RAGAS faithfulness and context precision; citation correctness rate (does the cited page support the verdict?) |

**Ablations**

- No retrieval vs. in-report evidence only vs. in-report + external evidence
- LLM numeric checking vs. Python numeric checking
- Small open model vs. API model

**Error analysis:** categorize 20–30 failure cases by cause, e.g. table parsing, retrieval miss, or the LLM over-trusting the company's wording.

## Timeline

The build takes ten weeks: the pipeline is complete by week 7, evaluation and the demo follow, and week 10 is for writing the paper.

&#91;embedded content: build schedule · 8 workstreams over 10 weeks\]

## Paper outline

The paper follows the pipeline, with the BRSR dataset and the claim taxonomy as its two headline contributions.

1. Introduction
2. Related Work: climate NLP, greenwashing detection, RAG fact-checking
3. Problem Formulation and Claim Taxonomy
4. Dataset: the BRSR corpus and annotation protocol
5. Method: the six-stage pipeline
6. Experiments
7. Results and Ablations
8. Error Analysis
9. Limitations and Ethics: false greenwashing accusations are a real harm, so frame outputs as risk flags for human auditors, not verdicts
10. Conclusion

**Target venues:** a climate-NLP workshop such as ClimateNLP at ACL, or NeurIPS Tackling Climate Change with ML.

## Sources

- [ClimRetrieve: A Benchmarking Dataset for Information Retrieval from Corporate Climate Disclosures (arXiv)](https://arxiv.org/pdf/2406.09818)
- [ClimRetrieve abstract (Swiss Finance Institute)](https://www.sfi.ch/en/publications/n-24-89-climretrieve-a-benchmarking-dataset-for-information-retrieval-from-corporate-climate-disclosures)
- [EmeraldData dataset card (HuggingFace)](https://huggingface.co/datasets/geoka/EmeraldData/blob/main/README.md)
- [EmeraldMind: A Knowledge Graph-Augmented Framework for Greenwashing Detection (arXiv)](https://arxiv.org/pdf/2512.11506v2)
- [NLP benchmark for corporate climate policy engagement (NeurIPS 2023)](https://papers.neurips.cc/paper_files/paper/2023/file/7ccaa4f9a89cce6619093226f26b84e6-Paper-Datasets_and_Benchmarks.pdf)
- [Benchmark dataset for greenhouse gas emission extraction (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12391291/)
- [AI turns the tide on greenwashing: the A3CG framework (NUS)](https://cde.nus.edu.sg/ai-turns-the-tide-on-greenwashing)
