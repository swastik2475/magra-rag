"""
retriever.py  -  FREE embeddings + FAISS (no C++ Build Tools needed on Windows).

ChromaDB → FAISS migration:
  - chromadb==1.0.0  still needed chroma-hnswlib (C++ Build Tools on some machines)
  - faiss-cpu ships with pre-built Windows wheels — just pip install, no compiler!
  - Embeddings: sentence-transformers (all-MiniLM-L6-v2) run locally, 100% free
  - Index persisted as two files: faiss_index.bin + faiss_meta.json
"""

from __future__ import annotations

import json
import os
import re
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_FILE  = "faiss_index.bin"
META_FILE   = "faiss_meta.json"
BATCH_SIZE  = 50


# ── Lazy-load embedding model ──────────────────────────────────────────────────

_model = None

def _get_model():
    """Load sentence-transformers model once (downloads ~90MB on first run)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[retriever] Loading embedding model '{EMBED_MODEL}' …")
        _model = SentenceTransformer(EMBED_MODEL)
        print("[retriever] Model ready.")
    return _model


def _embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts → numpy float32 array (N x D)."""
    model = _get_model()
    vecs  = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vecs.astype(np.float32)


# ── FAISS index helpers ────────────────────────────────────────────────────────

class FaissIndex:
    """
    Thin wrapper around a FAISS flat cosine index.

    Persisted as:
        <persist_dir>/faiss_index.bin  — the raw FAISS index
        <persist_dir>/faiss_meta.json  — list of chunk metadata (same order as vectors)
    """

    def __init__(self, dim: int):
        # IndexFlatIP on L2-normalised vectors = cosine similarity
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self.metas: list[dict] = []   # parallel list of chunk dicts (without embedding)

    def add(self, vecs: np.ndarray, metas: list[dict]) -> None:
        """Add L2-normalised vectors and their metadata."""
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self.metas.extend(metas)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[dict]:
        """Return top_k results sorted by cosine similarity (highest first)."""
        faiss.normalize_L2(query_vec)
        scores, idxs = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            meta = self.metas[idx]
            results.append({
                "id":       meta.get("id", str(idx)),
                "text":     meta.get("text", ""),
                "metadata": _safe_meta(meta),
                "distance": round(float(1.0 - score), 4),   # convert similarity → distance
                "citation": _format_citation(_safe_meta(meta)),
            })
        return results

    def count(self) -> int:
        return self.index.ntotal

    def save(self, persist_dir: str) -> None:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(Path(persist_dir) / INDEX_FILE))
        with open(Path(persist_dir) / META_FILE, "w") as f:
            json.dump(self.metas, f, default=str)
        print(f"[retriever] Saved FAISS index ({self.index.ntotal} vectors) → {persist_dir}")

    @classmethod
    def load(cls, persist_dir: str) -> "FaissIndex":
        idx_path  = Path(persist_dir) / INDEX_FILE
        meta_path = Path(persist_dir) / META_FILE
        if not idx_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"No FAISS index found in '{persist_dir}'. Run: python main.py --setup"
            )
        raw_index = faiss.read_index(str(idx_path))
        with open(meta_path) as f:
            metas = json.load(f)
        obj = cls.__new__(cls)
        obj.index = raw_index
        obj.metas = metas
        print(f"[retriever] Loaded FAISS index ({raw_index.ntotal} vectors) from {persist_dir}")
        return obj

    @classmethod
    def exists(cls, persist_dir: str) -> bool:
        return (
            (Path(persist_dir) / INDEX_FILE).exists()
            and (Path(persist_dir) / META_FILE).exists()
        )


# ── Build index ────────────────────────────────────────────────────────────────

def build_index(
    chunks: list[dict],
    persist_dir: str = "./faiss_db",
    force_rebuild: bool = False,
) -> FaissIndex:
    """
    Embed all chunks with free local model → store in FAISS.
    faiss-cpu works on Windows without C++ Build Tools.
    """
    if FaissIndex.exists(persist_dir) and not force_rebuild:
        print(f"[retriever] Index already exists in '{persist_dir}' — skipping rebuild.")
        print("[retriever] Use force_rebuild=True or --force-rebuild to regenerate.")
        return FaissIndex.load(persist_dir)

    print(f"[retriever] Embedding {len(chunks)} chunks locally (FREE) …")

    # Determine embedding dimension from a test batch
    sample_vecs = _embed([chunks[0]["text"]])
    dim = sample_vecs.shape[1]
    faiss_idx = FaissIndex(dim=dim)

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="  Embedding"):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vecs  = _embed(texts)
        faiss_idx.add(vecs, batch)

    faiss_idx.save(persist_dir)
    print(f"[retriever] Done — {faiss_idx.count()} docs indexed.")
    return faiss_idx


def get_index(persist_dir: str = "./faiss_db") -> FaissIndex:
    """Load existing FAISS index from disk."""
    return FaissIndex.load(persist_dir)


# ── Metadata helpers ───────────────────────────────────────────────────────────

def _safe_meta(chunk: dict) -> dict:
    """Flatten metadata to str/int/float."""
    return {
        "source":      str(chunk.get("source", "")),
        "chunk_type":  str(chunk.get("chunk_type", "")),
        "page_num":    int(chunk.get("page_num", 0)),
        "chunk_idx":   str(chunk.get("chunk_idx", "")),
        "activity_id": str(chunk.get("activity_id", "")),
        "line_item":   str(chunk.get("line_item", "")),
    }


def _format_citation(meta: dict) -> str:
    ct  = meta.get("chunk_type", "")
    src = meta.get("source", "")
    if ct == "schedule_activity":
        return f"[Schedule: Activity {meta.get('activity_id', '')} – {src}]"
    elif ct == "sov_line_item":
        return f"[SOV: {meta.get('line_item', '')} – {src}]"
    elif ct in ("pdf_text", "pdf_table"):
        return f"[Contract p.{meta.get('page_num', '?')} – {src}]"
    return f"[{src}]"


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    index: FaissIndex,
    top_k: int = 8,
    chunk_lookup: Optional[dict] = None,
) -> list[dict]:
    """
    Retrieve top_k relevant chunks.
    Step 1: embed query locally (free)
    Step 2: cosine search in FAISS
    Step 3: keyword boost for Activity IDs / SOV terms
    """
    if index.count() == 0:
        return []

    n = min(top_k, index.count())
    query_vec = _embed([query])
    chunks = index.search(query_vec, top_k=n)

    # Keyword boost
    if chunk_lookup:
        boosted = _keyword_boost(query, chunk_lookup)
        seen    = {c["id"] for c in chunks}
        chunks  = [b for b in boosted if b["id"] not in seen] + chunks

    return chunks


def _keyword_boost(query: str, chunk_lookup: dict) -> list[dict]:
    """Exact-match retrieval for Activity IDs (e.g. PROC-11) and SOV terms."""
    boosted = []

    # Activity ID patterns: PROC-11, MOB-3, MS-4, MRNDR-1 etc.
    ids = re.findall(r"\b([A-Z]{2,8}-\d{1,3})\b", query.upper())
    for aid in ids:
        for c in chunk_lookup.values():
            if c.get("activity_id", "").upper() == aid:
                boosted.append(_to_result(c))

    # SOV keyword boost
    sov_keywords = [
        "force main", "culvert", "earthwork", "concrete", "demolition",
        "storm", "marine drive", "allowance", "contingency", "plug valve",
        "gravity sewer", "general conditions", "overflow", "retainage",
    ]
    ql = query.lower()
    for kw in sov_keywords:
        if kw in ql:
            for c in chunk_lookup.values():
                if c.get("chunk_type") == "sov_line_item" and kw in c["text"].lower():
                    if not any(b["id"] == c["id"] for b in boosted):
                        boosted.append(_to_result(c))

    return boosted


def _to_result(chunk: dict) -> dict:
    meta = _safe_meta(chunk)
    return {
        "id":       chunk["id"],
        "text":     chunk["text"],
        "metadata": meta,
        "distance": 0.0,
        "citation": _format_citation(meta),
    }


def load_chunk_lookup(json_path: str = "./data/corpus_chunks.json") -> dict:
    """Load chunk JSON cache into an id-keyed dict."""
    with open(json_path) as f:
        chunks = json.load(f)
    return {c["id"]: c for c in chunks}
