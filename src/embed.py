# src/embed.py
"""
Embedding helper supporting:
 - SentenceTransformers (local dev)
 - OpenAI (cloud) with compatibility for openai<1.0 and openai>=1.0

embed_texts(texts) -> np.ndarray of shape (n, dim), dtype float32, L2-normalized.
"""

from __future__ import annotations
import os
from typing import List
import numpy as np

# Try local sentence-transformers first (preferred for dev)
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAVE_ST = True
except Exception:
    _HAVE_ST = False

# Try openai client (may be >=1.0 or older)
_HAVE_OPENAI = False
_OPENAI_NEW = False
try:
    import openai  # type: ignore
    # detect new SDK style: openai.OpenAI exists in >=1.0
    if hasattr(openai, "OpenAI"):
        _OPENAI_NEW = True
    _HAVE_OPENAI = True
except Exception:
    _HAVE_OPENAI = False

_ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_ST_MODEL = None

def _get_st_model(name: str = _ST_MODEL_NAME):
    global _ST_MODEL
    if _ST_MODEL is None:
        _ST_MODEL = SentenceTransformer(name)
    return _ST_MODEL

def _openai_embed_newapi(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Use OpenAI Python >=1.0 interface:
      from openai import OpenAI
      client = OpenAI()
      resp = client.embeddings.create(...)
    """
    # import lazily
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    embs = []
    for i in range(0, len(texts), 16):
        chunk = texts[i:i+16]
        resp = client.embeddings.create(model=model, input=chunk)
        for item in resp.data:
            embs.append(item.embedding)
    return np.asarray(embs, dtype="float32")

def _openai_embed_oldapi(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Use OpenAI python <1.0 interface:
      openai.api_key = ...
      resp = openai.Embedding.create(...)
    """
    import openai
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    embs = []
    for i in range(0, len(texts), 16):
        chunk = texts[i:i+16]
        resp = openai.Embedding.create(model=model, input=chunk)
        for item in resp["data"]:
            embs.append(item["embedding"])
    return np.asarray(embs, dtype="float32")

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Return L2-normalized float32 embeddings for `texts`.
    Priority:
      1) SentenceTransformers (if installed)
      2) OpenAI new SDK (if OPENAI_API_KEY set and openai>=1.0)
      3) OpenAI old SDK (if available)
    Raises RuntimeError if no backend available.
    """
    if not texts:
        return np.empty((0, 0), dtype="float32")

    # 1) Local ST (dev)
    if _HAVE_ST:
        model = _get_st_model()
        vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")

    # 2) OpenAI new SDK
    if _HAVE_OPENAI and _OPENAI_NEW and os.environ.get("OPENAI_API_KEY"):
        vecs = _openai_embed_newapi(texts)
    # 3) OpenAI old style
    elif _HAVE_OPENAI and not _OPENAI_NEW and os.environ.get("OPENAI_API_KEY"):
        vecs = _openai_embed_oldapi(texts)
    else:
        raise RuntimeError("No embedding backend available. Install sentence-transformers locally "
                           "or set OPENAI_API_KEY in env/secrets and ensure openai package is installed.")

    # Normalize to unit length for cosine similarity with FAISS IndexFlatIP
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    return np.asarray(vecs, dtype="float32")
