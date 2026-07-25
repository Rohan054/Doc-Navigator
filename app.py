"""Document Navigator - Streamlit demo.

Deploy: push app.py + rag_core.py + requirements.txt (+ optional pdfs/) to GitHub,
then share.streamlit.io -> New app -> main file app.py.
"""
import shutil, tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import rag_core as rc

st.set_page_config(page_title="Document Navigator", layout="wide")

# Visual badge per gate decision: (label, colour, one-line explanation).
BADGE = {"answer":  ("Answered", "#1a7f37", "Evidence cleared the confidence threshold."),
         "clarify": ("Needs clarification", "#9a6700", "Top two sources scored almost identically."),
         "refuse":  ("Refused", "#b42318", "No chunk cleared the confidence threshold.")}

PDF_DIR = Path("pdfs")


# Cache the built index so it is not rebuilt on every interaction; the `stamp`
# arg is part of the cache key, so changing PDFs or settings forces a rebuild.
@st.cache_resource(show_spinner="Building index...")
def build(pdf_dir: str, embedder: str, chunk_mode: str, title_prefix: bool, stamp: str):
    cfg = rc.Config(embedder=embedder, chunk_mode=chunk_mode, title_prefix=title_prefix)
    return rc.build_index(cfg, pdf_dir)


st.title("Document Navigator")
st.caption("A RAG assistant that shows *why* it answered: retrieval traces, citations, "
           "and an explicit confidence gate.")

with st.sidebar:
    # --- Document source: upload PDFs, or use any bundled in pdfs/ ---
    st.header("Documents")
    uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded:
        tmp = Path(tempfile.gettempdir()) / "docnav_pdfs"
        if st.button("Index uploaded PDFs", type="primary"):
            shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
            for f in uploaded:
                (tmp / f.name).write_bytes(f.getbuffer())
            st.session_state["pdf_dir"] = str(tmp)
            st.session_state["stamp"] = ",".join(sorted(f.name for f in uploaded))
            st.cache_resource.clear()
    if PDF_DIR.exists() and any(PDF_DIR.glob("*.pdf")):
        if st.button("Use bundled PDFs"):
            st.session_state["pdf_dir"] = str(PDF_DIR)
            st.session_state["stamp"] = "bundled"
            st.cache_resource.clear()

    # --- Live retrieval controls; every widget feeds the Config below ---
    st.header("Retrieval")
    embedder = st.selectbox("Embedder", ["minilm", "tfidf", "tfidf_char"], 0)
    chunk_mode = st.selectbox("Chunking", ["statement", "fixed", "whole"], 0,
                              help="Whole-document chunking caps precision@5 at 1/k.")
    title_prefix = st.checkbox("Prefix chunks with document title", True)
    mode = st.radio("Mode", ["hybrid", "dense", "bm25"], 0, horizontal=True)
    dense_weight = st.slider("Dense weight", 0.0, 1.0, 0.6, 0.05, disabled=(mode != "hybrid"))
    top_k = st.slider("top-k", 1, 10, 5)

    # --- Thresholds for the refuse / clarify logic ---
    st.header("Confidence gate")
    refuse_below = st.slider("Refuse below (cosine)", 0.0, 0.9, 0.35, 0.01)
    clarify_margin = st.slider("Clarify if margin below", 0.0, 0.3, 0.05, 0.01)

# Resolve which PDFs to index: user choice, else bundled, else prompt and stop.
pdf_dir = st.session_state.get("pdf_dir")
if not pdf_dir:
    if PDF_DIR.exists() and any(PDF_DIR.glob("*.pdf")):
        pdf_dir, st.session_state["stamp"] = str(PDF_DIR), "bundled"
    else:
        st.info("Upload PDFs in the sidebar, then press **Index uploaded PDFs**.")
        st.stop()

try:
    index = build(pdf_dir, embedder, chunk_mode, title_prefix,
                  st.session_state.get("stamp", "bundled"))
except Exception as exc:
    st.error(f"Could not build the index: {exc}")
    st.stop()

# Rebuild a Config from the live sidebar widgets for this query.
cfg = rc.Config(embedder=embedder, chunk_mode=chunk_mode, title_prefix=title_prefix,
                mode=mode, dense_weight=dense_weight, top_k=top_k,
                refuse_below=refuse_below, clarify_margin=clarify_margin)

# Corpus summary metrics.
c1, c2, c3 = st.columns(3)
c1.metric("PDFs", len(index.sources()))
c2.metric("Chunks", len(index))
c3.metric("Embedding dim", index.faiss_index.d)
with st.expander("Indexed files"):
    st.write(index.sources())

query = st.text_input("Ask a question", placeholder="e.g. What is the return window?")

if query:
    # Retrieve, gate, and answer in one call; res carries the full trace.
    res = rc.answer_question(index, query, cfg, mode=mode, top_k=top_k,
                             dense_weight=dense_weight)
    tr = res.trace
    label, colour, hint = BADGE[res.decision]
    st.markdown(
        f"<div style='padding:10px 14px;border-left:5px solid {colour};"
        f"background:rgba(0,0,0,0.04);border-radius:4px'>"
        f"<b style='color:{colour}'>{label}</b> &nbsp; confidence <code>{tr.confidence:.3f}</code>"
        f" &nbsp; margin <code>{tr.margin:+.3f}</code><br>"
        f"<span style='font-size:0.85em;opacity:0.75'>{hint}</span></div>",
        unsafe_allow_html=True)

    st.subheader("Answer")
    st.write(res.text)
    if res.citations:
        st.markdown("**Citations:** " + "  ".join(f"`{c}`" for c in res.citations))
    st.caption(f"Generator: {res.generator}")

    # Transparency: show every retrieved chunk and its scores.
    st.subheader("Retrieval trace")
    st.dataframe(pd.DataFrame([h.to_dict() for h in tr.hits]).set_index("rank"),
                 use_container_width=True)
    with st.expander("Retrieved chunks in full"):
        for h in tr.hits:
            st.markdown(f"**{h.rank}. `{h.chunk_id}`** - fused `{h.fused_score:.3f}` / "
                        f"cosine `{h.dense_score:.3f}` / bm25 `{h.bm25_score:.2f}`")
            st.write(h.text); st.divider()
    with st.expander("Trace as JSON"):
        st.json(res.to_dict())
