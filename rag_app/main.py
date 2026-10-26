"""CLI entry point for hierarchical document-intelligence RAG.

Usage:
    python main.py --pdf ./doc.pdf
    python main.py --pdf ./doc1.pdf ./doc2.pdf --ask "Compare revenue trends"

In the chat loop:
    /reset    clear conversation history
    /debug    toggle retrieved-context visibility
    /k N      change top-k for retrieval
    /quit     exit
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src import embeddings as _embeddings, llm as _llm
from src.chunker import CHUNKER_VERSION, chunk_pdf
from src.enrichment import enrich_chunks
from src.hierarchical import HierarchicalRetriever, MultiDocumentRetriever
from src.ingest import extract_pdf
from src.intelligence import DocumentIntelligenceRAG
from src.retriever import HybridRetriever
from src.summarizer import (
    build_document_summary,
    build_page_summaries,
    save_summaries,
    summaries_to_chunks,
)


console = Console()
ROOT = Path(__file__).parent
STORAGE = ROOT / "storage"


def _index_dir_for(pdf_path: Path) -> Path:
    backend = _embeddings.active_backend()
    model_tag = _embeddings.active_model().replace("/", "_")
    key = f"{pdf_path.resolve().as_posix()}|{backend}|{model_tag}|{CHUNKER_VERSION}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return STORAGE / f"{pdf_path.stem}-{backend}-{digest}"


def _doc_id_for(pdf_path: Path) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", pdf_path.stem).strip("_")[:40] or "document"
    digest = hashlib.sha1(pdf_path.resolve().as_posix().encode()).hexdigest()[:8]
    return f"{safe_stem}-{digest}"


def _build_hierarchical(pdf_path: Path, idx_dir: Path) -> HierarchicalRetriever:
    doc_id = _doc_id_for(pdf_path)
    console.print(f"[bold]Ingesting[/bold] {pdf_path.name} as [cyan]{doc_id}[/cyan] ...")
    pages = extract_pdf(pdf_path, doc_id=doc_id)
    console.print(f"  -> {len(pages)} pages")

    console.print("[bold]Chunking + enriching[/bold] ...")
    chunks = enrich_chunks(chunk_pdf(pdf_path, pages), pages)
    console.print(
        f"  -> {len(chunks)} chunks "
        f"({sum(1 for c in chunks if c.kind == 'table')} table-chunks)"
    )

    console.print("[bold]Precomputing hierarchical summaries[/bold] ...")
    page_summaries = build_page_summaries(pages, chunks)
    doc_summary = build_document_summary(doc_id, pages, page_summaries)
    save_summaries(idx_dir / "summaries.json", page_summaries, doc_summary)
    page_summary_chunks, document_summary_chunks = summaries_to_chunks(page_summaries, doc_summary)
    console.print(
        f"  -> {len(page_summary_chunks)} page summaries, "
        f"{len(document_summary_chunks)} document summary"
    )

    console.print(
        f"[bold]Embedding + indexing[/bold] "
        f"(backend={_embeddings.active_backend()}, model={_embeddings.active_model()}) ..."
    )
    retriever = HierarchicalRetriever.build(
        doc_id,
        chunks=chunks,
        page_summary_chunks=page_summary_chunks,
        document_summary_chunks=document_summary_chunks,
    )
    retriever.save(idx_dir)
    console.print(f"  -> saved to {idx_dir}")
    return retriever


def build_or_load(pdf_path: Path, rebuild: bool = False) -> HybridRetriever:
    """Backward-compatible single-document chunk index used by existing tests."""
    idx_dir = _index_dir_for(pdf_path)
    if not rebuild and HybridRetriever.index_exists(idx_dir):
        console.print(f"[dim]Loading cached index from {idx_dir}[/dim]")
        return HybridRetriever.load(idx_dir)
    return _build_hierarchical(pdf_path, idx_dir).chunk_index


def build_or_load_hierarchical(pdf_path: Path, rebuild: bool = False) -> HierarchicalRetriever:
    idx_dir = _index_dir_for(pdf_path)
    doc_id = _doc_id_for(pdf_path)
    if not rebuild and HierarchicalRetriever.index_exists(idx_dir):
        console.print(f"[dim]Loading cached hierarchical index from {idx_dir}[/dim]")
        return HierarchicalRetriever.load(doc_id, idx_dir)
    return _build_hierarchical(pdf_path, idx_dir)


def render_debug(turn) -> None:
    if getattr(turn, "intent", ""):
        console.print(f"[dim]Intent:[/dim] [cyan]{turn.intent}[/cyan]")
    if turn.rewritten and turn.rewritten.strip().lower() != turn.question.strip().lower():
        console.print(f"[dim]Rewritten query:[/dim] [italic]{turn.rewritten}[/italic]")

    if not turn.hits:
        console.print("[yellow]No retrieved context.[/yellow]")
        return

    tbl = Table(title="Retrieved context (top-k)", show_lines=False, header_style="bold cyan")
    tbl.add_column("Citation", style="green", no_wrap=True)
    tbl.add_column("Kind", no_wrap=True)
    tbl.add_column("Dense", justify="right")
    tbl.add_column("BM25", justify="right")
    tbl.add_column("RRF", justify="right")
    tbl.add_column("Snippet", overflow="fold")
    for hit in turn.hits:
        snippet = hit.chunk.text.replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:220].rstrip() + " ..."
        tbl.add_row(
            hit.chunk.citation,
            hit.chunk.kind,
            f"{hit.dense_score:.3f}",
            f"{hit.sparse_score:.2f}",
            f"{hit.score:.4f}",
            snippet,
        )
    console.print(tbl)


def chat_loop(rag: DocumentIntelligenceRAG, debug: bool) -> None:
    console.print(Panel.fit(
        "[bold]Hierarchical Document-Intelligence RAG[/bold]\n"
        f"LLM:    {_llm.active_backend()} - {_llm.active_model()}\n"
        f"Embed:  {_embeddings.active_backend()} - {_embeddings.active_model()}\n"
        f"Docs:   {', '.join(rag.retriever.doc_ids)}\n"
        "Commands: /reset  /debug  /k N  /quit",
        border_style="cyan",
    ))
    while True:
        try:
            question = console.input("\n[bold green]you[/bold green] > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye")
            return
        if not question:
            continue
        if question in ("/quit", "/exit"):
            return
        if question == "/reset":
            rag.reset()
            console.print("[dim]history cleared[/dim]")
            continue
        if question == "/debug":
            debug = not debug
            console.print(f"[dim]debug = {debug}[/dim]")
            continue
        if question.startswith("/k "):
            try:
                rag.top_k = max(1, int(question.split()[1]))
                console.print(f"[dim]top_k = {rag.top_k}[/dim]")
            except Exception:
                console.print("[red]usage: /k 10[/red]")
            continue

        try:
            turn = rag.ask(question)
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")
            continue

        if debug:
            render_debug(turn)

        console.print(Panel(Markdown(turn.answer), title="answer", border_style="magenta"))


def main() -> int:
    load_dotenv(ROOT / ".env")
    load_dotenv()

    parser = argparse.ArgumentParser(description="Hierarchical multi-document document-intelligence RAG")
    parser.add_argument("--pdf", required=True, nargs="+", help="Path(s) to one or more PDF documents")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild indexes from scratch")
    parser.add_argument("--k", type=int, default=10, help="Top-k retrieved evidence items")
    parser.add_argument("--no-debug", action="store_true", help="Hide retrieved context")
    parser.add_argument("--ask", help="Run a single question and exit")
    args = parser.parse_args()

    pdf_paths = [Path(value).expanduser().resolve() for value in args.pdf]
    missing = [path for path in pdf_paths if not path.exists()]
    if missing:
        console.print(f"[red]PDF not found:[/red] {missing[0]}")
        return 2

    try:
        retrievers = [build_or_load_hierarchical(path, rebuild=args.rebuild) for path in pdf_paths]
    except Exception as exc:
        console.print(f"[red]Failed to build index:[/red] {exc}")
        return 1

    rag = DocumentIntelligenceRAG(
        retriever=MultiDocumentRetriever(retrievers),
        top_k=args.k,
    )
    debug = not args.no_debug

    if args.ask:
        turn = rag.ask(args.ask)
        if debug:
            render_debug(turn)
        console.print(Panel(Markdown(turn.answer), title="answer", border_style="magenta"))
        return 0

    chat_loop(rag, debug=debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
