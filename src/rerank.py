# src/rerank.py
"""
Re-ranking utilities using a cross-encoder model.

A cross-encoder (like `ms-marco-MiniLM-L-6-v2`) is used to rescore candidate
documents given a query. Unlike bi-encoder embeddings (used in FAISS), the
cross-encoder jointly encodes (query, passage) pairs for more precise ranking.

Workflow:
  1. Embed query & candidates via bi-encoder (FAISS).
  2. Take top-N candidates.
  3. Use a cross-encoder to rescore relevance.
  4. Sort results by cross-encoder score (higher = more relevant).
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional

import numpy as np

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except Exception:
    CROSS_ENCODER_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Recommended small cross-encoder model
_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Cache the model globally (lazy load)
_classifier: Optional["CrossEncoder"] = None


def _get_cross_encoder(model_name: str = _DEFAULT_MODEL) -> Optional["CrossEncoder"]:
    """
    Lazy-load and cache a CrossEncoder model.

    Args:
        model_name: Hugging Face model identifier.

    Returns:
        CrossEncoder instance, or None if unavailable.
    """
    global _classifier
    if _classifier is None:
        if not CROSS_ENCODER_AVAILABLE:
            logger.warning("CrossEncoder not available; falling back to FAISS scores.")
            return None
        logger.info("Loading CrossEncoder model: %s", model_name)
        _classifier = CrossEncoder(model_name)
    return _classifier


def rerank_with_cross_encoder(
    query: str,
    candidates: List[Dict[str, str]],
    model_name: str = _DEFAULT_MODEL,
    top_k: Optional[int] = None,
    batch_size: int = 16,
) -> List[Dict[str, str]]:
    """
    Re-rank candidate documents using a cross-encoder model.

    Args:
        query: user query string
        candidates: list of candidate dicts with at least
            {"id","title","text","score"}
        model_name: Hugging Face model name for cross-encoder
        top_k: optional cutoff; return only top_k docs
        batch_size: batch size for prediction (avoid OOM on large candidate sets)

    Returns:
        List of candidate dicts sorted by 'rerank_score' (float).
    """
    if not candidates:
        return []

    ce = _get_cross_encoder(model_name)

    # Prepare query–document pairs
    pairs = [(query, (c.get("title") or "") + " . " + (c.get("text") or "")) for c in candidates]

    if ce is None:
        # Fallback: reuse original FAISS scores
        for c in candidates:
            c["rerank_score"] = float(c.get("score", 0.0))
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

    # Predict in batches (for memory efficiency)
    scores: List[float] = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        batch_scores = ce.predict(batch)
        scores.extend(batch_scores.tolist() if isinstance(batch_scores, np.ndarray) else batch_scores)

    # Attach scores
    for cand, score in zip(candidates, scores):
        cand["rerank_score"] = float(score)

    # Sort by rerank_score descending
    sorted_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    return sorted_candidates[:top_k] if top_k else sorted_candidates
