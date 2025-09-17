# src/embed.py
"""
Embedding helper with dual-mode:
 - Local mode: use SentenceTransformers (if installed)
 - Cloud mode: fallback to OpenAI embeddings when OPENAI_API_KEY is configured

The function embed_texts(...) always returns float32 L2-normalized vectors
compatible with FAISS inner-product index (cosine similarity).
"""

from __future__ import annotations
import os
import numpy as np
from typing import List

# Try to import sentence-transformers (may be absent on cloud)
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAVE_ST = True
except Exception:
    _HAVE_ST = False

# Optional OpenAI fallback (lightweight, pure-Python)
try:
    import openai
    _HAVE_OPENAI = True
except Exception:
    _HAVE_OPENAI = False

# Local cached model
_ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_ST_MODEL = None

def _get_st_model(name: str = _ST_MODEL_NAME):
    global _ST_MODEL
    if _ST_MODEL is None:
        _ST_MODEL = SentenceTransformer(name)
    return _ST_MODEL

def _openai_embed(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Call OpenAI embeddings API. Returns numpy array shape (n, dim).
    Requires OPENAI_API_KEY in env or Streamlit secrets.
    Uses text-embedding-3-small (dimension 1536) by default — FAISS dimension must match.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set for OpenAI embedding fallback.")
    if not _HAVE_OPENAI:
        raise RuntimeError("openai package not installed.")
    openai.api_key = api_key

    # call API in batches to avoid hitting length limits
    embs = []
    for i in range(0, len(texts), 16):
        chunk = texts[i : i + 16]
        resp = openai.Embedding.create(model=model, input=chunk)
        for item in resp["data"]:
            embs.append(item["embedding"])
    return np.asarray(embs, dtype="float32")

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed list of texts into float32 L2-normalized vectors.

    Priority:
      1) SentenceTransformers (if installed) -> dims 384 (default model)
      2) OpenAI embeddings if OPENAI_API_KEY is present -> dims depend on model (e.g. 1536)
    """
    if not texts:
        # return empty array with shape (0, dim) — caller should handle
        return np.empty((0, 0), dtype="float32")

    # 1) Local ST model (preferred for dev)
    if _HAVE_ST:
        model = _get_st_model()
        vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")

    # 2) OpenAI fallback when running in cloud (requires key)
    if os.environ.get("OPENAI_API_KEY") and _HAVE_OPENAI:
        vecs = _openai_embed(texts)
        # normalize to unit length (L2) for cosine similarity with FAISS inner-product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        return np.asarray(vecs, dtype="float32")

    # 3) No embedding backend available
    raise RuntimeError(
        "No embedding backend available. Install sentence-transformers locally or set OPENAI_API_KEY for cloud fallback."
    )
