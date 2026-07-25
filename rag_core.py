"""Document Navigator - transparent RAG core (single module).

Pipeline: PDFs -> clean -> chunk -> embed -> FAISS + BM25 -> hybrid fuse
          -> retrieval trace -> confidence gate -> cited extractive answer.

Written to disk by the notebook (%%writefile) and imported by both the notebook
and app.py, so there is exactly one copy of the pipeline.
"""
from __future__ import annotations

import os, re, csv, json, pickle, unicodedata
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

# Identical disclaimer line in every sample PDF -> pure index noise, dropped at chunk time.
BOILERPLATE = re.compile(r"this pdf is synthetic and intended for", re.I)
NUMBERED = re.compile(r"^\s*\d+[.)]\s+")   # "1. " / "2) " list markers (fixed/legacy use)
SENTENCE = re.compile(r"(?<=[.!?])\s+")    # split after . ! ? followed by whitespace
TOKEN = re.compile(r"[a-z0-9]+")           # BM25 tokenizer pattern


# ───────────────────────── config ─────────────────────────
@dataclass
class Config:
    """Every tunable in one place. Copy + tweak fields to run an experiment."""
    name: str = "default"
    chunk_mode: str = "sentence"      # sentence | fixed | whole
    chunk_size: int = 100             # words per chunk (fixed mode only)
    chunk_overlap: int = 20           # word overlap between chunks (fixed mode only)
    title_prefix: bool = True         # prepend doc title to each chunk's embedded text
    drop_boilerplate: bool = True     # remove the repeated disclaimer line
    embedder: str = "minilm"          # minilm | tfidf | tfidf_char
    mode: str = "hybrid"              # dense | bm25 | hybrid
    dense_weight: float = 0.6         # dense vs BM25 weight when mode="hybrid"
    top_k: int = 5                    # how many chunks to retrieve
    refuse_below: float = 0.35        # refuse if top chunk's cosine is under this
    clarify_margin: float = 0.05      # ask to clarify if top-2 are this close

    def to_dict(self) -> dict: return asdict(self)


# ───────────────────────── ingest ─────────────────────────
@dataclass
class Page:
    """One PDF page. source + page drive the [file:page] citation."""
    source: str; page: int; text: str; title: str


def clean_text(t: str) -> str:
    """Normalise extracted text: strip soft hyphens, de-hyphenate line breaks,
    collapse runs of spaces and blank lines."""
    t = t.replace("\u00ad", "")           # soft hyphen
    t = re.sub(r"-\n(?=\w)", "", t)        # join words split across a line break
    t = re.sub(r"[ \t]+", " ", t)          # collapse spaces/tabs
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def load_pages(pdf_dir: str | Path = "pdfs") -> list[Page]:
    """Read every PDF in pdf_dir into cleaned per-page Page objects."""
    from pypdf import PdfReader
    paths = sorted(Path(pdf_dir).glob("*.pdf"))
    if not paths:
        raise FileNotFoundError(f"No PDFs in {pdf_dir}/ — upload some first.")
    pages = []
    for p in paths:
        for i, pg in enumerate(PdfReader(str(p)).pages, start=1):   # pages are 1-indexed for citations
            txt = clean_text(pg.extract_text() or "")
            if not txt:
                continue
            # First non-empty line is treated as the page/document title.
            title = next((l.strip() for l in txt.split("\n") if l.strip()), p.stem)
            pages.append(Page(p.name, i, txt, title))
    if not pages:   # PDFs opened but yielded no text -> almost always scanned images
        raise RuntimeError("PDFs found but no extractable text (scanned images need OCR).")
    return pages


def corpus_stats(pages: list[Page]) -> dict:
    """Quick sanity numbers: how many files/pages/words you actually loaded."""
    w = sum(len(p.text.split()) for p in pages)
    return {"files": len({p.source for p in pages}), "pages": len(pages),
            "words": w, "avg_words_per_page": round(w / max(len(pages), 1), 1)}


# ───────────────────────── chunking ─────────────────────────
@dataclass
class Chunk:
    """A retrievable unit. `text` is shown to the user; `embed_text` is what gets
    embedded (may carry a title prefix for extra context)."""
    chunk_id: str; source: str; page: int; text: str; embed_text: str; title: str
    def citation(self) -> str: return f"[{self.source}:{self.page}]"


def _segments(pg: Page, cfg: Config) -> list[str]:
    """Split one page's text into raw segments per the chosen chunk_mode."""
    lines = [l.strip() for l in pg.text.split("\n") if l.strip()]
    if cfg.drop_boilerplate:
        lines = [l for l in lines if not BOILERPLATE.search(l)]

    if cfg.chunk_mode == "whole":         # one chunk per page (caps precision@k at 1/k)
        return ["\n".join(lines)] if lines else []

    if cfg.chunk_mode == "fixed":         # sliding window of `chunk_size` words
        words = " ".join(lines).split()
        step = max(cfg.chunk_size - cfg.chunk_overlap, 1)
        out = [" ".join(words[i:i + cfg.chunk_size]) for i in range(0, len(words), step)]
        return [s for s in out if len(s.split()) > 5]

    if cfg.chunk_mode == "sentence":      # one chunk per sentence (default)
        body = [l for l in lines if l != pg.title]        # drop the title line
        text = " ".join(body)
        segs = SENTENCE.split(text)                       # split . ! ? -> sentences
        segs = [s.strip() for s in segs if len(s.split()) >= 3]   # drop tiny fragments
        # Real-world PDFs can have very long sentences; cap at ~220 words.
        out = []
        for s in segs:
            w = s.split()
            if len(w) <= 220: out.append(s)
            else: out += [" ".join(w[i:i + 180]) for i in range(0, len(w), 150)]  # 30-word overlap
        return out

    raise ValueError(f"Unknown chunk_mode: {cfg.chunk_mode}")


def chunk_pages(pages: list[Page], cfg: Config) -> list[Chunk]:
    """Turn pages into Chunk objects, assigning stable ids and embed_text."""
    chunks = []
    for pg in pages:
        for j, seg in enumerate(_segments(pg, cfg)):
            chunks.append(Chunk(
                f"{pg.source}::p{pg.page}::c{j:02d}",                 # stable, human-readable id
                pg.source, pg.page, seg,
                f"{pg.title}. {seg}" if cfg.title_prefix else seg,    # title prefix aids retrieval
                pg.title))
    return chunks


# ───────────────────────── embedding ─────────────────────────
def _norm(v) -> np.ndarray:
    """L2-normalise rows so FAISS inner product == cosine similarity."""
    v = np.asarray(v, dtype="float32")
    if v.ndim == 1: v = v[None, :]
    d = np.linalg.norm(v, axis=1, keepdims=True); d[d == 0] = 1e-9
    return (v / d).astype("float32")


class MiniLM:
    """Dense semantic embeddings via sentence-transformers (default, local, no key)."""
    name = "minilm"
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name; self._load()
    def _load(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name, device="cpu")
        self.dim = self.model.get_sentence_embedding_dimension()
    def fit(self, corpus): return self          # nothing to fit; kept for a uniform interface
    def encode(self, texts):
        return _norm(self.model.encode(list(texts), batch_size=32, show_progress_bar=False))
    # Persist only the model NAME, not the ~90 MB torch weights, in the pickle.
    def __getstate__(self): return {"model_name": self.model_name, "dim": self.dim}
    def __setstate__(self, s): self.model_name = s["model_name"]; self._load()


class TfidfSVD:
    """Lexical embeddings via TF-IDF + SVD. Zero download, so it doubles as the
    offline fallback and a baseline.

    char_ngrams=True adds character n-grams, fixing morphology blindness
    ("return" vs "returned") but also raising similarity for OFF-corpus queries,
    which shrinks the answerable/unanswerable gap the refuse gate relies on.
    Retune refuse_below if you enable it.
    """
    name = "tfidf"
    def __init__(self, dim=128, char_ngrams=False):
        self.target_dim = dim; self.char_ngrams = char_ngrams; self.vec = self.svd = None
    def fit(self, corpus):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        word = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        if self.char_ngrams:
            from sklearn.pipeline import FeatureUnion   # combine word + character features
            self.vec = FeatureUnion([("w", word),
                ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                      sublinear_tf=True, min_df=2))])
        else:
            self.vec = word
        X = self.vec.fit_transform(corpus)
        k = int(min(self.target_dim, max(2, min(X.shape) - 1)))   # SVD dims can't exceed the data
        self.svd = TruncatedSVD(n_components=k, random_state=0).fit(X); self.dim = k
        return self
    def encode(self, texts):
        return _norm(self.svd.transform(self.vec.transform(list(texts))))


def get_embedder(name: str):
    """Factory. MiniLM falls back to TF-IDF if the model can't be downloaded."""
    if name == "minilm":
        try: return MiniLM()
        except Exception as e:
            print(f"[warn] MiniLM unavailable ({e}); using tfidf."); return TfidfSVD()
    if name == "tfidf": return TfidfSVD()
    if name == "tfidf_char": return TfidfSVD(char_ngrams=True)
    raise ValueError(name)


# ───────────────────────── index ─────────────────────────
def tokenize(t: str) -> list[str]: return TOKEN.findall(t.lower())   # shared by BM25 build + query


@dataclass
class Index:
    """Bundles everything a query needs: chunks, dense FAISS index, BM25, embedder."""
    cfg: Config; chunks: list[Chunk]; faiss_index: Any; bm25: Any; embedder: Any
    def __len__(self): return len(self.chunks)
    def sources(self): return sorted({c.source for c in self.chunks})

    def save(self, out="artifacts"):
        """Persist to disk. FAISS index + a pickle of the rest + a readable manifest."""
        import faiss
        o = Path(out); o.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(o / "faiss.index"))
        pickle.dump({"cfg": self.cfg, "chunks": self.chunks, "bm25": self.bm25,
                     "embedder": self.embedder}, open(o / "store.pkl", "wb"))
        (o / "manifest.json").write_text(json.dumps(
            {"n_chunks": len(self.chunks), "dim": self.faiss_index.d,
             "sources": self.sources(), "config": self.cfg.to_dict()}, indent=2))
        return o

    @classmethod
    def load(cls, out="artifacts"):
        """Reload a saved index (used by the deployed app to skip rebuilding)."""
        import faiss
        o = Path(out)
        if not (o / "faiss.index").exists():
            raise FileNotFoundError(f"No index in {o}/ — build one first.")
        s = pickle.load(open(o / "store.pkl", "rb"))
        return cls(s["cfg"], s["chunks"], faiss.read_index(str(o / "faiss.index")),
                   s["bm25"], s["embedder"])


def build_index(cfg: Config | None = None, pdf_dir="pdfs") -> Index:
    """Full build: load -> chunk -> embed -> FAISS (dense) + BM25 (sparse)."""
    import faiss
    from rank_bm25 import BM25Okapi
    cfg = cfg or Config()
    chunks = chunk_pages(load_pages(pdf_dir), cfg)
    if not chunks: raise RuntimeError("Chunking produced no chunks.")
    corpus = [c.embed_text for c in chunks]
    emb = get_embedder(cfg.embedder); emb.fit(corpus)     # fit is a no-op for MiniLM
    vecs = emb.encode(corpus)
    # IndexFlatIP = exact inner-product search; on normalised vectors that's cosine.
    fi = faiss.IndexFlatIP(vecs.shape[1]); fi.add(np.ascontiguousarray(vecs))
    return Index(cfg, chunks, fi, BM25Okapi([tokenize(t) for t in corpus]), emb)


# ───────────────────────── retrieval + trace ─────────────────────────
@dataclass
class Hit:
    """One retrieved chunk with its scores — the row-level unit of a trace."""
    rank: int; chunk_id: str; source: str; page: int; text: str
    dense_score: float; bm25_score: float; fused_score: float
    def citation(self): return f"[{self.source}:{self.page}]"
    def to_dict(self):
        return {"rank": self.rank, "chunk_id": self.chunk_id, "source": self.source,
                "page": self.page, "dense_score": round(self.dense_score, 4),
                "bm25_score": round(self.bm25_score, 4),
                "fused_score": round(self.fused_score, 4), "text": self.text}


@dataclass
class Trace:
    """The transparency object: query + ranked hits + derived confidence signals."""
    query: str; mode: str; top_k: int
    hits: list[Hit] = field(default_factory=list); meta: dict = field(default_factory=dict)

    @property
    def top_score(self):
        """Ranking score of the best hit. NOT for gating: hybrid min-max pins it to 1.0."""
        return self.hits[0].fused_score if self.hits else 0.0

    @property
    def confidence(self):
        """Raw cosine of the top hit. Comparable across queries, so this is what gates."""
        return round(self.hits[0].dense_score, 4) if self.hits else 0.0

    @property
    def margin(self):
        """Cosine gap between rank 1 and rank 2. Small gap => two docs competing."""
        return round(self.hits[0].dense_score - self.hits[1].dense_score, 4) if len(self.hits) > 1 else self.confidence

    def to_dict(self):
        return {"query": self.query, "mode": self.mode, "top_k": self.top_k,
                "confidence": self.confidence, "margin": self.margin,
                "hits": [h.to_dict() for h in self.hits], **self.meta}

    def dataframe(self):
        """Trace as a pandas table for display in the notebook / app."""
        import pandas as pd
        return pd.DataFrame([h.to_dict() for h in self.hits]).set_index("rank")


def _minmax(x):
    """Scale a score array to 0..1 so dense and BM25 become comparable before fusing."""
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


def search(index: Index, query: str, top_k=None, mode=None, dense_weight=None) -> Trace:
    """Retrieve top_k chunks and return a full Trace. mode selects the scorer."""
    cfg = index.cfg
    top_k = top_k or cfg.top_k; mode = mode or cfg.mode
    w = cfg.dense_weight if dense_weight is None else dense_weight
    n = len(index.chunks)

    # Dense: cosine similarity of the query against every chunk (search all n, sort later).
    scores, ids = index.faiss_index.search(np.ascontiguousarray(index.embedder.encode([query])), n)
    dense = np.zeros(n, dtype="float32"); dense[ids[0]] = scores[0]
    # Sparse: BM25 lexical overlap score per chunk.
    bm25 = np.asarray(index.bm25.get_scores(tokenize(query)), dtype="float32")

    if mode == "dense": fused = dense
    elif mode == "bm25": fused = _minmax(bm25)
    elif mode == "hybrid":
        # Min-max each per query before the weighted sum: cosine and BM25 live on
        # different scales, so raw addition would let BM25 dominate.
        fused = w * _minmax(dense) + (1 - w) * _minmax(bm25)
    else: raise ValueError(mode)

    order = np.argsort(-fused)[:top_k]     # indices of the top_k highest fused scores
    hits = [Hit(r + 1, index.chunks[i].chunk_id, index.chunks[i].source, index.chunks[i].page,
                index.chunks[i].text, float(dense[i]), float(bm25[i]), float(fused[i]))
            for r, i in enumerate(order)]
    return Trace(query, mode, top_k, hits, {"dense_weight": w, "n_chunks": n})


# ───────────────────────── answering ─────────────────────────
Decision = Literal["answer", "clarify", "refuse"]
REFUSAL = ("I don't have enough evidence in these documents to answer that. "
           "The closest material found is shown in the trace below.")


@dataclass
class Answer:
    """Final response: the text, the gate decision, citations, and its trace."""
    text: str; decision: Decision
    citations: list[str] = field(default_factory=list)
    trace: Trace | None = None; generator: str = "extractive"
    def to_dict(self):
        return {"answer": self.text, "decision": self.decision, "citations": self.citations,
                "generator": self.generator, "trace": self.trace.to_dict() if self.trace else None}


def gate(trace: Trace, cfg: Config) -> Decision:
    """Decide answer / clarify / refuse from the trace's confidence and margin."""
    if not trace.hits or trace.confidence < cfg.refuse_below:
        return "refuse"                       # nothing similar enough -> don't guess
    if (trace.margin < cfg.clarify_margin and len(trace.hits) > 1
            and trace.hits[0].source != trace.hits[1].source):
        return "clarify"                      # two different docs tied -> ask which
    return "answer"


def _extractive(trace: Trace, n=1):
    """Build an answer from the top n chunks verbatim, with de-duplicated citations."""
    parts, cites, seen = [], [], set()
    for h in trace.hits[:n]:
        parts.append(f"{h.text.rstrip('.')}. {h.citation()}")
        if h.citation() not in seen: seen.add(h.citation()); cites.append(h.citation())
    return " ".join(parts), cites


def answer_question(index: Index, query: str, cfg: Config | None = None, **kw) -> Answer:
    """Top-level entry point: retrieve -> gate -> answer/clarify/refuse, all cited."""
    cfg = cfg or index.cfg
    trace = search(index, query, top_k=kw.pop("top_k", cfg.top_k),
                   mode=kw.pop("mode", cfg.mode), **kw)
    d = gate(trace, cfg)
    if d == "refuse":
        return Answer(REFUSAL, d, [], trace, "gate")
    if d == "clarify":
        a, b = trace.hits[0], trace.hits[1]
        return Answer(f"That could refer to either {a.source} or {b.source} — they scored "
                      f"almost identically (gap {trace.margin:.3f}). Which did you mean?",
                      d, [], trace, "gate")
    t, c = _extractive(trace)                 # confident enough -> quote the evidence
    return Answer(t, "answer", c, trace, "extractive")
