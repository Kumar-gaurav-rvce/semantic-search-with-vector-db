# src/indexer.py
"""
FAISS index and document-id helper utilities.

This module provides small, well-documented helpers to:
 - create a FAISS index suitable for cosine-similarity (inner-product on normalized vectors)
 - save / load the index from disk
 - persist and load document id lists (keeps FAISS positions aligned with doc ids)
 - append new vectors + metadata into the index + SQLite metadata table

Notes:
 - We assume embeddings are normalized (L2) prior to insertion. The embedder in this repo
   already returns normalized vectors. If you cannot guarantee that, call
   faiss.normalize_L2(vectors) before adding.
 - For production systems you may want to store raw embeddings separately (e.g. as .npy
   per-shard) so the index can be rebuilt without re-embedding.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss
import os
import requests

# local utils: ensures sqlite table exists and returns DB connection
from .utils import ensure_metadata_table, get_conn

# Default file paths (project root)
INDEX_PATH = Path("faiss_index.bin")
DOC_IDS_PATH = Path("doc_ids.npy")

# Default embedding dimensionality used by the model in this repo
DIM = 384  # match sentence-transformers/all-MiniLM-L6-v2


# ------------------------
# Index creation / helpers
# ------------------------
def create_index(dim: int = DIM) -> faiss.Index:
    """
    Create a FAISS index configured for cosine similarity search.

    We use IndexFlatIP (inner product) and expect vectors to be L2-normalized.
    For normalized vectors, inner-product == cosine similarity.

    Args:
        dim (int): embedding dimensionality.

    Returns:
        faiss.Index: a newly created FAISS index instance.
    """
    return faiss.IndexFlatIP(dim)

def _maybe_download(path: Path, url_env: str):
    url = os.environ.get(url_env)
    if url and not path.exists():
        # stream download
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        tmp = str(path) + ".part"
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        Path(tmp).rename(path)

def save_index(index: faiss.Index, path: Path = INDEX_PATH) -> None:
    """
    Persist FAISS index to disk.

    Args:
        index (faiss.Index): index to save.
        path (Path): destination file path.
    """
    faiss.write_index(index, str(path))


def load_index(path: Path = INDEX_PATH, dim: int = DIM) -> faiss.Index:
    """
    Load FAISS index from disk, or create a new empty one if path missing.

    Args:
        path (Path): file location to read index from.
        dim (int): dimensionality to use when creating a new index.

    Returns:
        faiss.Index: loaded or newly created index.
    """
    _maybe_download(path, "INDEX_URL")
    if not Path(path).exists():
        # Return a fresh index; caller should persist after adding vectors
        return create_index(dim)
    return faiss.read_index(str(path))


# ------------------------
# Doc ID persistence
# ------------------------
def save_doc_ids(ids: List[str], path: Path = DOC_IDS_PATH) -> None:
    """
    Save the list of document ids to a numpy file.

    Args:
        ids (List[str]): list of doc ids in the same order as FAISS vectors.
        path (Path): destination npy file path.
    """
    np.save(str(path), np.array(ids, dtype=object), allow_pickle=True)


def load_doc_ids(path: Path = DOC_IDS_PATH) -> List[str]:
    """
    Load document ids from disk. Returns empty list if file not present.

    Args:
        path (Path): location of saved doc ids.

    Returns:
        List[str]: doc ids in insertion order (matching FAISS idx positions).
    """
    _maybe_download(path, "DOC_IDS_URL")   # ✅ use `path` not `doc_ids_path`
    if not Path(path).exists():
        return []
    loaded = np.load(str(path), allow_pickle=True)
    # ensure we return a native python list of strings
    return list(loaded.tolist())



# ------------------------
# Adding vectors + metadata
# ------------------------
def add_vectors(
    index: faiss.Index,
    ids: List[str],
    vectors: np.ndarray,
    metas: List[Dict],
    index_path: Path = INDEX_PATH,
    doc_ids_path: Path = DOC_IDS_PATH,
) -> None:
    """
    Append vectors to FAISS index, persist doc ids and metadata.

    Args:
        index (faiss.Index): in-memory FAISS index instance.
        ids (List[str]): document ids (strings). Length must equal vectors.shape[0].
        vectors (np.ndarray): float32 array shape (N, DIM). Expected L2-normalized.
        metas (List[Dict]): list of metadata dicts (e.g. title, text, source, created_at).
        index_path (Path): where to persist the FAISS file after adding.
        doc_ids_path (Path): where to persist doc ids list.

    Raises:
        AssertionError: if input lengths mismatch.
        ValueError: if vector dtype/dimension mismatch.
    """
    # Basic validation

    assert len(ids) == vectors.shape[0] == len(metas), "ids, vectors, metas length mismatch"

    # Ensure numpy dtype and two-dimensionality
    if vectors.dtype != np.float32:
        vectors = vectors.astype("float32")

    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D vectors array, got shape {vectors.shape}")

    # Inferred incoming vector dimensionality
    incoming_dim = vectors.shape[1]

    # If index has a declared dimension, ensure it matches the incoming vectors.
    # For FAISS IndexFlatIP, the attribute is accessible via index.d
    try:
        index_dim = int(getattr(index, "d", getattr(index, "dim", None)))
    except Exception:
        index_dim = None

    if index_dim is not None:
        if index_dim != incoming_dim:
            raise ValueError(f"FAISS index expects vectors of dim {index_dim}, but got vectors with dim {incoming_dim}")
    else:
        # If index has no declared dim (unlikely), we proceed but warn
        # (no-op here; FAISS indices typically have attribute d)
        pass

    # Sanity check: if using IP index, vectors should be normalized for cosine similarity.
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        # Normalize for safety (prevents incorrect similarity results)
        vectors = vectors / norms[:, None]

    # Now append vectors to the index (FAISS requires matching dimensionality)
    index.add(vectors)


    # Persist/append doc ids to disk, keeping insertion order aligned with FAISS positions
    existing_ids = load_doc_ids(doc_ids_path)
    combined_ids = existing_ids + ids
    save_doc_ids(combined_ids, doc_ids_path)

    # Persist metadata into sqlite table 'docs' (id, title, text, source, created_at)
    ensure_metadata_table()
    conn = get_conn()
    cur = conn.cursor()
    for i, meta in enumerate(metas):
        cur.execute(
            "INSERT OR REPLACE INTO docs (id, title, text, source, created_at) VALUES (?,?,?,?,?)",
            (
                ids[i],
                meta.get("title"),
                meta.get("text"),
                meta.get("source", "unknown"),
                meta.get("created_at"),
            ),
        )
    conn.commit()
    conn.close()

    # Save the updated FAISS index to disk
    save_index(index, index_path)


# ------------------------
# Index rebuild (placeholder)
# ------------------------
def rebuild_index_from_meta(
    index_path: Path = INDEX_PATH, doc_ids_path: Path = DOC_IDS_PATH, dim: int = DIM
) -> None:
    """
    Placeholder: rebuild FAISS index from stored embeddings.

    Many production pipelines persist raw embeddings (e.g. .npy files). If you have
    persisted embeddings, implement this function to read them and reconstruct the index.

    Raises:
        NotImplementedError: if called in this demo.
    """
    raise NotImplementedError("Rebuild requires stored embeddings. Persist embeddings to rebuild index.")


# ------------------------
# Convenience getter
# ------------------------
def get_index_and_doc_ids(
    index_path: Path = INDEX_PATH, doc_ids_path: Path = DOC_IDS_PATH, dim: int = DIM
) -> Tuple[faiss.Index, List[str]]:
    """
    Load (or create) FAISS index and the corresponding doc_id list.

    Returns:
        (index, doc_ids): index may be empty if no vectors persisted yet.
    """
    index = load_index(index_path, dim)
    doc_ids = load_doc_ids(doc_ids_path)
    return index, doc_ids
