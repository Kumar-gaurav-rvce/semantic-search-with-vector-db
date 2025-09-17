# src/summary.py
"""
Robust extractive summarizer with safe fallbacks.

Priority:
  1. If sentence-transformers is available, use it to embed sentences and pick
     centroid-nearest sentences (extractive).
  2. Otherwise use a TF-IDF + sentence-vector centroid method (scikit-learn).
  3. (Optional) If openai package is available and OPENAI_API_KEY is set, you
     can enable an abstractive summarization call (commented out below).

This file intentionally avoids importing heavy HF libs at module import time.
"""
from __future__ import annotations

from typing import List
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Try lazy imports
_HAVE_ST = False
_have_sklearn = False
_have_openai = False

try:
    # don't import at top-level; assign flag only
    import importlib

    if importlib.util.find_spec("sentence_transformers") is not None:
        _HAVE_ST = True
except Exception:
    _HAVE_ST = False

try:
    # scikit-learn provides TF-IDF vectorizer used in fallback method
    if importlib.util.find_spec("sklearn") is not None:
        _have_sklearn = True
except Exception:
    _have_sklearn = False

try:
    if importlib.util.find_spec("openai") is not None:
        _have_openai = True
except Exception:
    _have_openai = False

# Default model name (if sentence-transformers available locally)
_SUMMARY_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_summary_st_model = None


def _get_st_model():
    """Lazy-load SentenceTransformer only when needed and available."""
    global _summary_st_model
    if not _HAVE_ST:
        raise RuntimeError("sentence-transformers not available")
    if _summary_st_model is None:
        from sentence_transformers import SentenceTransformer  # local import
        _summary_st_model = SentenceTransformer(_SUMMARY_ST_MODEL)
    return _summary_st_model


def _safe_sent_tokenize(text: str) -> List[str]:
    """Try nltk sent_tokenize if available, else fall back to regex."""
    try:
        import nltk

        if nltk is not None:
            try:
                return nltk.tokenize.sent_tokenize(text)
            except Exception:
                pass
    except Exception:
        pass

    import re

    parts = re.split(r"(?<=[\.\?\!])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _tfidf_centroid_summary(sentences: List[str], n_sentences: int = 3) -> str:
    """
    Fallback summarizer: compute TF-IDF vectors for sentences, take centroid,
    score sentences by cosine similarity to centroid and return top-N in original order.
    Requires scikit-learn.
    """
    if not sentences:
        return ""
    if len(sentences) <= n_sentences:
        return " ".join(sentences)

    if not _have_sklearn:
        # ultimate fallback: return first n_sentences
        return " ".join(sentences[:n_sentences])

    # local import to avoid requiring sklearn at module import time
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    try:
        vec = TfidfVectorizer(max_features=2000, stop_words="english")
        X = vec.fit_transform(sentences)  # shape (n_sent, n_features)
        Xn = normalize(X, norm="l2", axis=1)
        centroid = Xn.mean(axis=0)  # scipy sparse; convert to array
        # compute cosine similarity via dot product (centroid is 1 x f)
        sims = (Xn @ centroid.T).A1 if hasattr((Xn @ centroid.T), "A1") else (Xn @ centroid.T).ravel()
        top_idx = np.argsort(-sims)[:n_sentences]
        top_idx_sorted = sorted(top_idx)
        return " ".join([sentences[i] for i in top_idx_sorted])
    except Exception as e:
        logger.debug("TF-IDF summarizer failed: %s", e)
        return " ".join(sentences[:n_sentences])


def summarize_texts(texts: List[str], n_sentences: int = 3) -> str:
    """
    Public summarization function.

    Args:
        texts: list of document strings (e.g., top-K results)
        n_sentences: desired number of extractive sentences in final summary

    Returns:
        A short extractive summary string (never raises due to missing heavy deps).
    """
    if not texts:
        return ""

    # Merge inputs into single document
    doc = " ".join([t for t in texts if t])
    if not doc:
        return ""

    # Tokenize into sentences (safe)
    sentences = _safe_sent_tokenize(doc)
    if not sentences:
        return ""

    # 1) Prefer SentenceTransformer-based embedding centroid if available
    if _HAVE_ST:
        try:
            model = _get_st_model()
            # local import of numpy-friendly encode
            sent_embs = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
            # centroid and scores
            centroid = np.mean(sent_embs, axis=0, keepdims=True)
            sims = (sent_embs @ centroid.T).squeeze()
            top_idx = np.argsort(-sims)[:n_sentences]
            top_idx_sorted = sorted(top_idx)
            return " ".join([sentences[i] for i in top_idx_sorted])
        except Exception as e:
            logger.debug("ST summarizer failed, falling back: %s", e)

    # 2) TF-IDF centroid fallback (scikit-learn)
    try:
        return _tfidf_centroid_summary(sentences, n_sentences=n_sentences)
    except Exception as e:
        logger.debug("TF-IDF fallback failed: %s", e)

    # 3) Final fallback: return first n_sentences
    return " ".join(sentences[:n_sentences])
