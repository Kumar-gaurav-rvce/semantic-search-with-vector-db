# src/embed.py
"""
Embedding utilities for semantic search.

This module wraps a SentenceTransformer model (MiniLM by default)
and provides helper functions to embed text into normalized vectors
that can be stored in FAISS for semantic search.

Functions:
    get_model: Lazy-load the SentenceTransformer model.
    embed_texts: Convert a list of texts into normalized float32 embeddings.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Global cache for the embedding model
_MODEL = None


def get_model(name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Lazy-load and return the SentenceTransformer model.

    Args:
        name (str, optional): Hugging Face model name. Defaults to
            "sentence-transformers/all-MiniLM-L6-v2".

    Returns:
        SentenceTransformer: Loaded embedding model (cached globally).
    """
    global _MODEL
    if _MODEL is None:
        # Load once and cache globally (saves time & memory)
        _MODEL = SentenceTransformer(name)
    return _MODEL


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts into normalized vectors.

    Args:
        texts (list[str]): List of input documents/queries to embed.

    Returns:
        np.ndarray: 2D array of shape (n_texts, dim), dtype float32.
            Vectors are L2-normalized for cosine similarity search.
    """
    if not texts:
        return np.empty((0, get_model().get_sentence_embedding_dimension()), dtype="float32")

    model = get_model()

    # Generate embeddings (normalized for cosine similarity)
    vectors = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return np.asarray(vectors, dtype="float32")
