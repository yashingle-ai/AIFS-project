"""Run the 5 must-pass acceptance tests against the cached index.

Usage:
    python -m tests.acceptance
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import build_or_load  # noqa: E402
from src.chat import ConversationalRAG  # noqa: E402


PDF = Path(__file__).resolve().parent.parent.parent / "testing_files" / "AEL_Earnings_Presentation_Q2-FY26.pdf"


def header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def show_turn(turn) -> None:
    print(f"Q: {turn.question}")
    if turn.rewritten and turn.rewritten.strip().lower() != turn.question.strip().lower():
        print(f"   (rewritten → {turn.rewritten})")
    print("\nTop retrieved chunks:")
    for h in turn.hits[:5]:
        snip = h.chunk.text.replace("\n", " ")
        if len(snip) > 140:
            snip = snip[:140].rstrip() + " …"
        print(f"  {h.chunk.citation:<14} ({h.chunk.kind:5}) "
              f"d={h.dense_score:.3f} bm25={h.sparse_score:5.2f} rrf={h.score:.4f}  {snip}")
    print("\nAnswer:")
    print(turn.answer)


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    retriever = build_or_load(PDF, rebuild=False)
    rag = ConversationalRAG(retriever=retriever, top_k=6)

    # Test 1: grounded fact
    header("TEST 1 — Grounded fact (business segments)")
    show_turn(rag.ask("What are the major business segments discussed in the document?"))
    time.sleep(2)

    rag.reset()

    # Test 2: numeric
    header("TEST 2 — Numeric (consolidated total income H1-26)")
    show_turn(rag.ask("What is the consolidated total income in H1-26?"))
    time.sleep(2)

    rag.reset()

    # Test 3: cross-section
    header("TEST 3 — Cross-section (EBITDA drivers H1-26)")
    show_turn(rag.ask("What drivers are mentioned for EBITDA changes in H1-26?"))
    time.sleep(2)

    rag.reset()

    # Test 4: negative control
    header("TEST 4 — Negative control (CEO email)")
    show_turn(rag.ask("What is the CEO's email address?"))
    time.sleep(2)

    rag.reset()

    # Test 5: multi-turn follow-up
    header("TEST 5a — Conversational (airport summary)")
    show_turn(rag.ask("Summarize airport performance in H1-26."))
    time.sleep(2)
    header("TEST 5b — Follow-up (break that down into passenger and cargo)")
    show_turn(rag.ask("Break that down into passenger and cargo changes."))

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
