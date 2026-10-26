"""Lightweight eval harness for the RAG pipeline.

Runs a JSONL golden dataset through ConversationalRAG and computes:
  - Recall@k         : did at least one expected_page appear in top-k retrieval?
  - Page precision   : fraction of top-k pages that were "expected"
  - Refusal accuracy : did `refuse=true` items return "Not found...", and did
                       `refuse=false` items NOT return "Not found..."?
  - Faithfulness     : every must_contain string appears verbatim in the answer
                       (or any-of for must_contain_any)
  - Per-question PASS/FAIL printout + a final summary table.

Usage:
    python -m tests.eval --pdf ../testing_files/AEL_Earnings_Presentation_Q2-FY26.pdf \
                         --golden tests/golden_AEL.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.chat import ConversationalRAG  # noqa: E402
from src.chunker import CHUNKER_VERSION, chunk_pdf  # noqa: E402
from src.ingest import extract_pdf         # noqa: E402
from src.retriever import HybridRetriever  # noqa: E402


@dataclass
class Case:
    q: str
    expected_pages: list[int]
    must_contain: list[str]
    must_contain_any: list[str]
    refuse: bool


def load_golden(path: Path) -> list[Case]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        d = json.loads(line)
        cases.append(Case(
            q=d["q"],
            expected_pages=d.get("expected_pages", []),
            must_contain=d.get("must_contain", []),
            must_contain_any=d.get("must_contain_any", []),
            refuse=bool(d.get("refuse", False)),
        ))
    return cases


def build_or_load_index(pdf: Path) -> HybridRetriever:
    key = f"{pdf.resolve().as_posix()}|{CHUNKER_VERSION}"
    h = hashlib.sha1(key.encode()).hexdigest()[:12]
    cache = ROOT / "storage" / f"_eval-{pdf.stem}-{h}"
    if HybridRetriever.index_exists(cache):
        return HybridRetriever.load(cache)
    print(f"  building index → {cache}", flush=True)
    pages = extract_pdf(pdf)
    chunks = chunk_pdf(pdf, pages)
    r = HybridRetriever.build(chunks)
    r.save(cache)
    return r


def _bool(b): return "✓" if b else "✗"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--top-k", type=int, default=6)
    args = ap.parse_args()

    pdf = Path(args.pdf).resolve()
    golden = Path(args.golden).resolve()
    if not pdf.exists():    sys.exit(f"PDF not found: {pdf}")
    if not golden.exists(): sys.exit(f"Golden not found: {golden}")

    print(f"PDF: {pdf.name}")
    print(f"Golden: {golden.name}")
    cases = load_golden(golden)
    print(f"Cases: {len(cases)}")

    retriever = build_or_load_index(pdf)
    rag = ConversationalRAG(retriever=retriever, top_k=args.top_k)

    rows = []
    n_pass = 0
    t0 = time.monotonic()

    for i, c in enumerate(cases, 1):
        rag.reset()
        try:
            turn = rag.ask(c.q)
        except Exception as e:
            rows.append((i, c.q[:60], False, False, False, False, f"ERROR: {e}"))
            continue

        retrieved_pages = [h.chunk.page for h in turn.hits]
        recall_at_k = (not c.expected_pages) or any(p in retrieved_pages for p in c.expected_pages)
        precision = (len([p for p in retrieved_pages if p in c.expected_pages]) / len(retrieved_pages)
                     if retrieved_pages and c.expected_pages else None)

        ans_lower = turn.answer.lower()
        # Accept any canonical refusal phrase the LLM might produce
        _REFUSAL_PHRASES = (
            "not found in the document",
            "not explicitly mentioned",
            "not mentioned in the document",
            "no information",
            "cannot be found",
            "is not available",
            "is not provided",
            "is not mentioned",
            "not available in the",
            "not present in the",
        )
        is_refusal = any(p in ans_lower for p in _REFUSAL_PHRASES)

        # Refusal accuracy
        refusal_ok = (c.refuse == is_refusal)

        # Faithfulness: ALL must_contain must appear, OR any of must_contain_any
        faithful = True
        if not c.refuse:
            if c.must_contain:
                faithful = all(s.lower() in ans_lower for s in c.must_contain)
            elif c.must_contain_any:
                faithful = any(s.lower() in ans_lower for s in c.must_contain_any)
        else:
            faithful = is_refusal  # refusal cases pass faithfulness iff refused

        case_pass = recall_at_k and refusal_ok and faithful
        if case_pass:
            n_pass += 1

        precision_str = "—" if precision is None else f"{precision:.2f}"
        rows.append((i, c.q[:60], recall_at_k, refusal_ok, faithful, case_pass, precision_str))

        # Per-case detail
        print(f"\n[{i:>2}] {'PASS' if case_pass else 'FAIL'} — {c.q}")
        print(f"     retrieved pages: {retrieved_pages}   expected: {c.expected_pages or '—'}")
        print(f"     recall@k={_bool(recall_at_k)}  refusal_ok={_bool(refusal_ok)}  faithful={_bool(faithful)}")
        if c.must_contain:        print(f"     must_contain: {c.must_contain}")
        if c.must_contain_any:    print(f"     must_contain_any: {c.must_contain_any}")
        ans_short = turn.answer[:200].replace("\n", " ")
        print(f"     answer: {ans_short}{'…' if len(turn.answer) > 200 else ''}")

    dur = time.monotonic() - t0
    print("\n" + "=" * 80)
    print(f"{'#':>3}  {'recall':>6} {'refusal':>7} {'faith':>5} {'prec':>5}  question")
    print("-" * 80)
    for r in rows:
        i, q, rec, ref, fa, _, prec = r
        print(f"{i:>3}  {_bool(rec):>6} {_bool(ref):>7} {_bool(fa):>5} {prec:>5}  {q}")
    print("=" * 80)
    print(f"PASS: {n_pass}/{len(cases)}  ({n_pass/len(cases):.0%})   total time: {dur:.1f}s")
    return 0 if n_pass == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
