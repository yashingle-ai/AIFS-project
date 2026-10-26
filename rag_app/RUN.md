# How to run QuantX

Every command below is run from the `rag_app/` directory unless stated otherwise.

```bash
cd "d:\-PS2-QuantX-Financial-Document-Intelligence-RAG\rag_app"
```

---

## 1. One-time setup

**Create the virtual environment** (Python 3.11+):

```bash
python -m venv .venv
```

**Activate it:**

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat

# Git Bash / macOS / Linux
source .venv/Scripts/activate      # or .venv/bin/activate on macOS/Linux
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Create `.env`** in `rag_app/` with your keys:

```ini
# --- pick your provider -------------------------------------------------
LLM_BACKEND=nvidia            # nvidia | gemini
EMBED_BACKEND=nvidia          # nvidia | gemini
VISION_BACKEND=nvidia         # nvidia | gemini | none
RERANK_BACKEND=nvidia         # nvidia | none

# --- NVIDIA NIM (free at build.nvidia.com) ------------------------------
NVIDIA_API_KEY=nvapi-...

# --- Google Gemini (free at aistudio.google.com/apikey) -----------------
GEMINI_API_KEY=...

# --- optional -----------------------------------------------------------
HF_API_KEY=hf_...             # only for the Qwen summariser
INDEX_IMAGES=true             # false = skip vision captioning (faster indexing)
RAG_AUTH_TOKEN=               # set a value to require Bearer auth on /api/*
ALLOW_ORIGINS=                # csv of extra CORS origins
```

`.env` is gitignored. **Never commit it and never paste a key into chat.**

---

## 2. Run the web app  ← the main one

```bash
.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Then open:

```
http://127.0.0.1:8000
```

Add `--reload` while developing so the server restarts on file changes:

```bash
.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

**What you should see:** the DocChat interface — upload panel on the left,
conversation in the middle, retrieved-chunk inspector on the right. Drop a PDF
in, wait for the status chip to turn `ready`, then ask a question.

Indexing runs in the background. The document card shows live progress
(`extracting` → `describing images` → `chunking` → `summarizing` → `embedding`
→ `ready`). A 40-page deck takes a few minutes on first upload.

---

## 3. Verify it is working

```bash
# which models are active
curl http://127.0.0.1:8000/api/info

# what is indexed
curl http://127.0.0.1:8000/api/documents

# Prometheus metrics
curl http://127.0.0.1:8000/api/metrics
```

`/api/info` should return something like:

```json
{"llm_backend":"nvidia","llm_model":"meta/llama-3.3-70b-instruct",
 "embed_backend":"nvidia","embed_model":"nvidia/nv-embedqa-e5-v5",
 "rerank_backend":"nvidia","rerank_enabled":true}
```

---

## 4. Run the CLI (no browser)

Useful for debugging retrieval, because it prints the retrieved chunks.

```bash
# interactive session over one or more PDFs
.venv\Scripts\python.exe main.py --pdf ../testing_files/AEL_Earnings_Presentation_Q2-FY26.pdf

# multiple documents at once
.venv\Scripts\python.exe main.py --pdf doc1.pdf doc2.pdf

# one question, then exit
.venv\Scripts\python.exe main.py --pdf report.pdf --ask "What was total income in H1 FY26?"

# force a rebuild of the indexes
.venv\Scripts\python.exe main.py --pdf report.pdf --rebuild

# wider retrieval, hide the debug context
.venv\Scripts\python.exe main.py --pdf report.pdf --k 15 --no-debug
```

| Flag | Meaning |
|---|---|
| `--pdf` | one or more PDF paths (required) |
| `--rebuild` | rebuild indexes from scratch |
| `--k` | top-k evidence items (default 10) |
| `--ask` | single question, non-interactive |
| `--no-debug` | hide the retrieved context |

---

## 5. Run the Streamlit UI (alternative frontend)

```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

Opens on `http://localhost:8501`.

---

## 6. Run the evaluation harness

Scores retrieval recall, page precision, refusal accuracy and faithfulness
against the hand-labelled question set.

```bash
.venv\Scripts\python.exe -m tests.eval ^
    --pdf ../testing_files/AEL_Earnings_Presentation_Q2-FY26.pdf ^
    --golden tests/golden_AEL.jsonl
```

Acceptance tests (five must-pass scenarios, including a question that **must**
be refused):

```bash
.venv\Scripts\python.exe -m tests.acceptance
```

---

## 7. Useful development commands

```bash
# syntax-check every module
.venv\Scripts\python.exe -m compileall -q src/

# confirm all modules import
.venv\Scripts\python.exe -c "import src.intelligence, src.retriever, src.service; print('ok')"

# inspect a built index
.venv\Scripts\python.exe -c "import faiss; i=faiss.read_index('storage/<doc_id>/faiss.index'); print(i.ntotal, 'vectors x', i.d, 'dims')"

# wipe all indexed documents and start clean
rm -rf storage/
```

---

## 8. Environment variables reference

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BACKEND` | `nvidia` | `nvidia` \| `gemini` — answer generation |
| `EMBED_BACKEND` | `nvidia` | `nvidia` \| `gemini` — embeddings |
| `VISION_BACKEND` | `nvidia` | `nvidia` \| `gemini` \| `none` — chart captioning |
| `RERANK_BACKEND` | `nvidia` | `nvidia` \| `none` — cross-encoder reranking |
| `NVIDIA_API_KEY` | — | required when any backend is `nvidia` |
| `GEMINI_API_KEY` | — | required when any backend is `gemini` |
| `HF_API_KEY` | — | Qwen summariser only |
| `NVIDIA_CHAT_MODEL` | `meta/llama-3.3-70b-instruct` | override the chat model |
| `NVIDIA_EMBED_MODEL` | `nvidia/nv-embedqa-e5-v5` | override the embedding model |
| `NVIDIA_VISION_MODEL` | `meta/llama-3.2-11b-vision-instruct` | override the vision model |
| `GEMINI_CHAT_MODEL` | `gemini-2.5-flash` | override the chat model |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-001` | override the embedding model |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NIM endpoint |
| `INDEX_IMAGES` | `true` | `false` skips vision captioning |
| `RAG_AUTH_TOKEN` | *(empty)* | set to require `Authorization: Bearer <token>` |
| `ALLOW_ORIGINS` | localhost dev ports | csv of extra CORS origins |

---

## 9. Troubleshooting

**`NVIDIA_API_KEY not set`**
`.env` is missing or not in `rag_app/`. It must sit next to `server.py`.

**Document stuck on `indexing`**
Check the terminal running uvicorn — `_do_index` catches errors and writes them
into `meta.json`. Read `storage/<doc_id>/meta.json` and look at the `error` field.

**Port 8000 already in use**
Use another port: `--port 8010`.

**Answers are always "Not found in the document"**
The refusal floor is rejecting weak retrieval. Confirm the document status is
`ready`, then check the retrieved scores — the thresholds are `MIN_DENSE_SCORE`
and `MIN_SPARSE_SCORE` in `src/intelligence.py`.

**Indexing is very slow**
Vision captioning dominates. Set `INDEX_IMAGES=false` to skip it while you are
testing the text pipeline.

**Rate-limit / 429 errors**
Both backends back off and retry automatically. Repeated failures usually mean
the free-tier quota is exhausted — switch `*_BACKEND` to the other provider.

---

## 10. Quick reference

| I want to… | Command |
|---|---|
| Run the web app | `.venv\Scripts\python.exe -m uvicorn server:app --port 8000` |
| Run the CLI | `.venv\Scripts\python.exe main.py --pdf <file>` |
| Run Streamlit | `.venv\Scripts\python.exe -m streamlit run app.py` |
| Run the eval | `.venv\Scripts\python.exe -m tests.eval --pdf <file> --golden tests/golden_AEL.jsonl` |
| Run acceptance tests | `.venv\Scripts\python.exe -m tests.acceptance` |
| Check active models | `curl http://127.0.0.1:8000/api/info` |
| Start clean | `rm -rf storage/` |






---

## 11. COPY-PASTE — running this on your laptop

### A. First time only (fresh machine or fresh clone)

Open **PowerShell** and run these in order:

```powershell
# 1. go to the project
cd "d:\-PS2-QuantX-Financial-Document-Intelligence-RAG\rag_app"

# 2. create the virtual environment
python -m venv .venv

# 3. activate it
.\.venv\Scripts\Activate.ps1

# 4. install everything
pip install -r requirements.txt
```

If step 3 fails with *"running scripts is disabled on this system"*, run this
once and then retry:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**5. Create the `.env` file** — it must sit next to `server.py`:

```powershell
notepad .env
```

Paste this in, put your real key on the `NVIDIA_API_KEY` line, and save:

```ini
LLM_BACKEND=nvidia
EMBED_BACKEND=nvidia
VISION_BACKEND=nvidia
RERANK_BACKEND=nvidia
NVIDIA_API_KEY=your-nvidia-api-key
```

A free key comes from **https://build.nvidia.com**.

---

### B. Every time after that — just two commands

```powershell
cd "d:\-PS2-QuantX-Financial-Document-Intelligence-RAG\rag_app"
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000** in your browser.

Stop the server with **Ctrl+C** in that terminal.

> You do **not** need to activate the venv when you call
> `.\.venv\Scripts\python.exe` directly — the full path already selects the
> right interpreter. That is why the daily flow is two lines, not three.

---

### C. Same thing in Git Bash

```bash
cd "d:/-PS2-QuantX-Financial-Document-Intelligence-RAG/rag_app"
./.venv/Scripts/python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
```

---

### D. Check it actually started

In a **second** terminal:

```powershell
curl http://127.0.0.1:8000/api/info
```

Healthy output looks like:

```json
{"llm_backend":"nvidia","llm_model":"meta/llama-3.3-70b-instruct",
 "embed_backend":"nvidia","embed_model":"nvidia/nv-embedqa-e5-v5",
 "rerank_backend":"nvidia","rerank_enabled":true}
```

If that returns, the app is up. Open the browser URL and upload a PDF.

---

### E. Moving it to a different laptop

Copy the whole project folder **except** these — they are machine-specific and
will be regenerated:

```
.venv/          the virtual environment (rebuild with step A)
storage/        indexed documents (re-upload the PDFs)
__pycache__/    compiled bytecode
```

Then run section **A** on the new machine. The `.env` file has to be recreated
by hand, because it is gitignored and holds your keys.




cd "d:\-PS2-QuantX-Financial-Document-Intelligence-RAG\rag_app"
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000


---

## 12. Daily restart — after shutting the laptop down

Nothing is lost on shutdown. `.venv`, `.env` and `storage/` all persist, and the
indexes stay valid, so there is **no re-indexing and no re-install**.

### Start it

```powershell
cd "d:\-PS2-QuantX-Financial-Document-Intelligence-RAG\rag_app"
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. Stop with **Ctrl+C**.

### Health check (second terminal, 10 seconds)

```powershell
curl http://127.0.0.1:8000/api/info
curl http://127.0.0.1:8000/api/documents
```

Healthy output as of 2026-09-28:

```json
{"llm_backend":"nvidia","llm_model":"nvidia/nemotron-3-super-120b-a12b",
 "embed_backend":"nvidia","embed_model":"nvidia/nemotron-3-embed-1b",
 "rerank_backend":"none","rerank_enabled":false}
```

The document list should show `"status":"ready"`.

---

### If answers stop working: check for model retirement first

NVIDIA retires hosted models on a rolling basis and this project has already been
hit three times (reranker May 2026, embeddings Aug 2026, the whole Llama family
by Sep 2026). The symptom is an `HTTP 410 Gone` in the terminal.

**Step 1 — find what still works:**

```powershell
.\.venv\Scripts\python.exe -c "import os,httpx;from dotenv import load_dotenv;load_dotenv('.env');k=os.getenv('NVIDIA_API_KEY');H={'Authorization':f'Bearer {k}'};[print(m) for m in sorted(x['id'] for x in httpx.get('https://integrate.api.nvidia.com/v1/models',headers=H,timeout=60).json()['data'])]"
```

Being **listed does not mean callable** — many listed models return 404 or 410.
Always probe before configuring one.

**Step 2 — probe a candidate:**

```powershell
.\.venv\Scripts\python.exe -c "import os,httpx,time;from dotenv import load_dotenv;load_dotenv('.env');k=os.getenv('NVIDIA_API_KEY');t=time.perf_counter();r=httpx.post('https://integrate.api.nvidia.com/v1/chat/completions',headers={'Authorization':f'Bearer {k}'},json={'model':'MODEL_ID_HERE','messages':[{'role':'user','content':'ok'}],'max_tokens':5},timeout=40);print(r.status_code,int((time.perf_counter()-t)*1000),'ms',r.text[:120])"
```

**Step 3 — update `.env`** with the working model id and restart.

### If the EMBEDDING model changed, you must re-index

A chat-model swap needs nothing extra. An **embedding**-model swap invalidates
every stored index, because vectors are only comparable within the embedding
space that produced them — and the dimension usually differs too (the Aug 2026
replacement went 1024 → 2048).

The app detects this and refuses to serve a stale index rather than crashing:

```
index is stale and cannot be queried - re-upload the PDF to rebuild it
reason: indexed with model 'nvidia/nv-embedqa-e5-v5', now running 'nvidia/nemotron-3-embed-1b'
```

To rebuild, just **re-upload the PDF through the UI**. The old directory is
replaced automatically. A 4-page document takes about 30 seconds.

### Known-good configuration (2026-09-28)

```ini
NVIDIA_CHAT_MODEL=nvidia/nemotron-3-super-120b-a12b
NVIDIA_EMBED_MODEL=nvidia/nemotron-3-embed-1b
NVIDIA_VISION_MODEL=meta/llama-3.2-11b-vision-instruct
RERANK_BACKEND=none
```

Gemini remains configured as a fallback — set `LLM_BACKEND=gemini` and
`EMBED_BACKEND=gemini` if NVIDIA has an outage. Switching the embed backend also
requires a re-index.


cd "d:\-PS2-QuantX-Financial-Document-Intelligence-RAG\rag_app"
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000

