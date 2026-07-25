
import os, shutil, tempfile
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


def get_api_key():
    try: return st.secrets["OPENAI_API_KEY"]
    except Exception: return os.getenv("OPENAI_API_KEY")


API_KEY = get_api_key()
if API_KEY:
    os.environ["OPENAI_API_KEY"] = API_KEY



@st.cache_resource(show_spinner="Building index (embedding with OpenAI)...")
def build(pdf_dir: str, stamp: str, chunk_mode: str, chunk_size: int, chunk_overlap: int):
    # chunk_mode/size/overlap are part of the cache key, so changing them rebuilds.
    cfg = rc.Config(embedder="openai", chunk_mode=chunk_mode, title_prefix=True,
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return rc.build_index(cfg, pdf_dir)


st.title("Document Navigator")
st.caption("A RAG assistant that shows *why* it answered: retrieval traces, citations, "
           "and an explicit confidence gate. Powered by OpenAI.")


if not API_KEY:
    st.error("This app needs an OpenAI API key. Add **OPENAI_API_KEY** in "
             "Streamlit **Settings → Secrets** (or set it as an environment variable "
             "when running locally), then reload.")
    st.stop()

with st.sidebar:
    # Document source: upload PDFs, or use any bundled in pdfs
    st.header("Documents")
    uploaded = st.file_uploader("Select PDFs", type="pdf", accept_multiple_files=True)
    if uploaded:
        tmp = Path(tempfile.gettempdir()) / "docnav_pdfs"
        if st.button("Upload PDFs", type="primary"):
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

    # Chunking: sentence for small/clean docs, recursive for larger PDFs
    st.header("Chunking")
    chunk_mode = st.selectbox("Method", ["sentence", "recursive"], 0,
        help="sentence: one chunk per sentence (small, clean docs). "
             "recursive: LangChain RecursiveCharacterTextSplitter (larger PDFs).")
    if chunk_mode == "recursive":
        chunk_size = st.slider("Chunk size (characters)", 200, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, 200, 20)
    else:
        chunk_size, chunk_overlap = 100, 20   # unused by sentence mode

    # Retrieval controls
    st.header("Retrieval")
    top_k = st.slider("top-k", 1, 10, 5)

    # Thresholds for the refuse / clarify logic
    st.header("Confidence gate")
    refuse_below = st.slider("Refuse below (cosine)", 0.0, 0.9, 0.35, 0.01)
    clarify_margin = st.slider("Clarify if margin below", 0.0, 0.3, 0.05, 0.01)

# Resolve which PDFs to index: user choice, else bundled, else prompt and stop.
pdf_dir = st.session_state.get("pdf_dir")
if not pdf_dir:
    if PDF_DIR.exists() and any(PDF_DIR.glob("*.pdf")):
        pdf_dir, st.session_state["stamp"] = str(PDF_DIR), "bundled"
    else:
        st.info("Select PDFs in the sidebar, then press **Upload PDFs**.")
        st.stop()

try:
    index = build(pdf_dir, st.session_state.get("stamp", "bundled"),
                  chunk_mode, chunk_size, chunk_overlap)
except Exception as exc:
    st.error(f"Could not build the index: {exc}")
    st.stop()

# Pipeline config, plus the live retrieval/gate sliders for this query.
cfg = rc.Config(embedder="openai", chunk_mode=chunk_mode, title_prefix=True,
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                mode="hybrid", dense_weight=0.6, top_k=top_k,
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
    res = rc.answer_question(index, query, cfg, mode="hybrid", top_k=top_k,
                             dense_weight=0.6)
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
