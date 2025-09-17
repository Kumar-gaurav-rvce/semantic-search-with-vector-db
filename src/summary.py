# src/summary.py
"""
Extractive summarization utilities.

This module implements a lightweight extractive summarizer:
  1. Combine input texts into one "document"
  2. Split into sentences (NLTK if available, else regex fallback)
  3. Embed sentences with a SentenceTransformer
  4. Compute centroid embedding
  5. Select sentences closest to centroid as the summary

If the model is unavailable or something fails, it falls back to returning
the first N sentences.

This approach is useful for quick summarization of semantic search results.
"""

from __future__ import annotations

from typing import List
import numpy as np

try:
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize
    _HAVE_NLTK = True
except Exception:
    _HAVE_NLTK = False

from sentence_transformers import SentenceTransformer

# Default model for embeddings
_SUMMARY_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_summary_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the summarization embedding model."""
    global _summary_model
    if _summary_model is None:
        _summary_model = SentenceTransformer(_SUMMARY_MODEL_NAME)
    return _summary_model


def _safe_sent_tokenize(text: str) -> List[str]:
    """
    Tokenize into sentences with safety fallback.

    Args:
        text: raw text to split

    Returns:
        List of sentence strings.
    """
    if not text or not text.strip():
        return []

    if _HAVE_NLTK:
        try:
            sents = _nltk_sent_tokenize(text)
            if sents and len(" ".join(sents)) > 0:
                return sents
        except Exception:
            pass  # fall through to regex fallback

    # Regex fallback: split on sentence-ending punctuation
    import re
    parts = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def summarize_texts(texts: List[str], n_sentences: int = 3) -> str:
    """
    Summarize a list of texts using extractive centroid-based method.

    Args:
        texts: List of raw document strings.
        n_sentences: Number of summary sentences to return.

    Returns:
        Extractive summary string (concatenated sentences).
    """
    if not texts:
        return ""

    # Merge texts into one pseudo-document
    document = " ".join([t for t in texts if t])
    sentences = _safe_sent_tokenize(document)
    if not sentences:
        return ""

    # If doc is already short, return everything
    if len(sentences) <= n_sentences:
        return " ".join(sentences)

    try:
        # Embed all sentences
        model = _get_model()
        sent_embs = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)

        # Compute centroid of all sentence embeddings
        centroid = np.mean(sent_embs, axis=0, keepdims=True)

        # Score similarity of each sentence to centroid
        sims = (sent_embs @ centroid.T).squeeze()

        # Take top-N sentences (keep original order)
        top_idx = np.argsort(-sims)[:n_sentences]
        summary = " ".join([sentences[i] for i in sorted(top_idx)])
        return summary

    except Exception:
        # Fallback: just return the first n sentences
        return " ".join(sentences[:n_sentences])
