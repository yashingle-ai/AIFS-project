"""Streamlit UI for the RAG agent.

Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.chat import ConversationalRAG
from src.chunker import CHUNKER_VERSION, chunk_pdf
from src.ingest import extract_pdf
from src.retriever import HybridRetriever


load_dotenv()

ROOT = Path(__file__).parent
STORAGE = ROOT / "storage"
STORAGE.mkdir(exist_ok=True)


st.set_page_config(page_title="Doc-Grounded RAG Chat", page_icon="📄", layout="wide")
st.title("📄 Document-Grounded Conversational RAG")
st.caption("Hybrid retrieval (FAISS + BM25) · Gemini · grounded answers with citations")


def _index_dir_for(pdf_path: Path) -> Path:
    key = f"{pdf_path.resolve().as_posix()}|{CHUNKER_VERSION}"
    h = hashlib.sha1(key.encode()).hexdigest()[:12]
    return STORAGE / f"{pdf_path.stem}-{h}"


@st.cache_resource(show_spinner=False)
def build_retriever(pdf_bytes: bytes, name: str) -> HybridRetriever:
    # Write to a stable temp path so the index is cached per-PDF
    h = hashlib.sha1(pdf_bytes).hexdigest()[:12]
    tmp = Path(tempfile.gettempdir()) / f"rag_{h}_{name}"
    tmp.write_bytes(pdf_bytes)
    idx_dir = _index_dir_for(tmp)
    if HybridRetriever.index_exists(idx_dir):
        return HybridRetriever.load(idx_dir)
    pages = extract_pdf(tmp)
    chunks = chunk_pdf(tmp, pages)
    r = HybridRetriever.build(chunks)
    r.save(idx_dir)
    return r


with st.sidebar:
    st.header("Setup")
    api_key_in = st.text_input(
        "GEMINI_API_KEY",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Get one free at https://aistudio.google.com/apikey",
    )
    if api_key_in:
        os.environ["GEMINI_API_KEY"] = api_key_in

    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
    top_k = st.slider("Top-k retrieved chunks", 3, 12, 6)
    show_debug = st.toggle("Show retrieved chunks", value=True)

    if st.button("Reset chat"):
        st.session_state.pop("rag", None)
        st.session_state.pop("messages", None)
        st.rerun()


if not pdf_file:
    st.info("Upload a PDF in the sidebar to begin.")
    st.stop()

if not os.getenv("GEMINI_API_KEY"):
    st.warning("Add your Gemini API key in the sidebar.")
    st.stop()

with st.spinner("Building index (first time only) …"):
    retriever = build_retriever(pdf_file.getvalue(), pdf_file.name)

if "rag" not in st.session_state:
    st.session_state.rag = ConversationalRAG(retriever=retriever, top_k=top_k)
    st.session_state.messages = []  # list[(role, content, hits)]
else:
    st.session_state.rag.top_k = top_k
    st.session_state.rag.retriever = retriever

# replay history
for role, content, hits in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant" and hits and show_debug:
            with st.expander(f"🔎 Retrieved {len(hits)} chunks"):
                for h in hits:
                    st.markdown(
                        f"**{h.chunk.citation}** · _{h.chunk.kind}_ · "
                        f"dense={h.dense_score:.3f} · bm25={h.sparse_score:.2f} · "
                        f"rrf={h.score:.4f}"
                    )
                    st.code(h.chunk.text[:800] + ("…" if len(h.chunk.text) > 800 else ""))

q = st.chat_input("Ask a question about the document …")
if q:
    with st.chat_message("user"):
        st.markdown(q)
    st.session_state.messages.append(("user", q, []))
    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            turn = st.session_state.rag.ask(q)
        st.markdown(turn.answer)
        if show_debug:
            with st.expander(f"🔎 Retrieved {len(turn.hits)} chunks"):
                for h in turn.hits:
                    st.markdown(
                        f"**{h.chunk.citation}** · _{h.chunk.kind}_ · "
                        f"dense={h.dense_score:.3f} · bm25={h.sparse_score:.2f} · "
                        f"rrf={h.score:.4f}"
                    )
                    st.code(h.chunk.text[:800] + ("…" if len(h.chunk.text) > 800 else ""))
    st.session_state.messages.append(("assistant", turn.answer, turn.hits))
