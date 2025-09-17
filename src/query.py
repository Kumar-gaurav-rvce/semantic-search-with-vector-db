# src/query.py
"""
Main search pipeline for the semantic-search demo.

Pipeline steps:
  1. Optional query expansion (WordNet)
  2. Embed the (expanded) query
  3. Retrieve candidate documents from FAISS (k * 4)
  4. Load metadata for candidates from SQLite
  5. Compute fuzzy-match scores (RapidFuzz)
  6. Optional re-ranking with a Cross-Encoder
  7. Optional filtering by score threshold
  8. "Did you mean?" suggestions (RapidFuzz over a small title sample)
  9. Highlight matched terms in results
 10. Optional clustering of result embeddings
 11. Extractive summarization of top results
 12. Build score explanations for plotting

Returned structure:
{
  "results": [ {id,title,text,score,rerank_score,fuzzy_score,highlighted_text,cluster}, ... ],
  "suggestions": [ (candidate, score%), ... ],
  "summary": "extractive summary...",
  "explanations": [float score per result],
  "expanded_terms": [ ... ]
}
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

import numpy as np
from rapidfuzz import fuzz, process

# Local helpers
from .expand import expand_query
from .rerank import rerank_with_cross_encoder
from .clustering import cluster_results_by_embeddings
from .summary import summarize_texts
from .embed import embed_texts
from .indexer import get_index_and_doc_ids
from .utils import get_conn

# Module-level logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Default top-k if caller doesn't provide one
DEFAULT_TOP_K = 5


def _highlight_text(text: str, query_terms: List[str]) -> str:
    """
    Highlight query_terms in the given text using Markdown bold (**term**).
    Longer terms are applied first to avoid partial overlaps.

    Args:
        text: original document text
        query_terms: list of query + expansion strings

    Returns:
        Markdown-friendly string with matched terms bolded.
    """
    if not text:
        return ""

    # sort terms by length desc so we replace multi-word phrases before single words
    unique_terms = sorted({t for t in query_terms if t and t.strip()}, key=lambda s: -len(s))
    out = text
    for term in unique_terms:
        # case-insensitive replacement using regex
        try:
            pattern = re.compile(re.escape(term), flags=re.IGNORECASE)
            out = pattern.sub(lambda m: f"**{m.group(0)}**", out)
        except re.error:
            # if the term contains regex-problematic characters, skip highlighting for it
            logger.debug("Skipping highlight for term with regex error: %s", term)
            continue
    return out


def did_you_mean(query: str, candidates: List[str], limit: int = 3) -> List[Tuple[str, int]]:
    """
    Suggest similar candidate strings using RapidFuzz.

    Args:
        query: user query
        candidates: list of candidate strings (e.g., titles)
        limit: maximum suggestions to return

    Returns:
        List of (candidate, score_percent) where score >= 50.
    """
    if not candidates:
        return []
    results = process.extract(query, candidates, scorer=fuzz.token_sort_ratio, limit=limit)
    suggestions = [(r[0], int(r[1])) for r in results if r[1] >= 50]
    return suggestions


def _fetch_metadata_for_ids(doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch metadata (id, title, text) for given doc_ids from the SQLite 'docs' table.

    Args:
        doc_ids: list of document id strings

    Returns:
        Mapping doc_id -> metadata dict
    """
    if not doc_ids:
        return {}

    conn = get_conn()
    cur = conn.cursor()
    placeholders = ",".join(["?"] * len(doc_ids))
    query = f"SELECT id, title, text FROM docs WHERE id IN ({placeholders})"
    cur.execute(query, tuple(doc_ids))
    rows = cur.fetchall()
    conn.close()

    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        out[r["id"]] = {"id": r["id"], "title": r["title"], "text": r["text"]}
    return out


def _compute_fuzzy_scores(query: str, texts: List[str]) -> List[float]:
    """
    Compute a normalized 0..1 fuzzy matching score for each text against the query.

    Uses RapidFuzz token_sort_ratio and partial_ratio and takes the maximum.

    Args:
        query: user query
        texts: list of strings to compare

    Returns:
        list of floats in [0.0, 1.0]
    """
    out: List[float] = []
    for t in texts:
        if not t:
            out.append(0.0)
            continue
        score = max(fuzz.token_sort_ratio(query, t), fuzz.partial_ratio(query, t))
        out.append(float(score) / 100.0)
    return out


def search(
    query: str,
    k: int = DEFAULT_TOP_K,
    min_score: float = 0.0,
    use_expansion: bool = True,
    rerank: bool = True,
    cluster: bool = True,
) -> Dict[str, Any]:
    """
    Perform a full semantic search pipeline.

    Args:
        query: the user's input query string
        k: number of final results to return
        min_score: minimum rerank_score (or raw score if rerank missing) to include
        use_expansion: whether to apply WordNet-based expansion
        rerank: whether to apply cross-encoder re-ranking
        cluster: whether to cluster the final result set

    Returns:
        Dict with keys: results, suggestions, summary, explanations, expanded_terms
    """
    if not query or not query.strip():
        return {"results": [], "suggestions": [], "summary": "", "explanations": [], "expanded_terms": []}

    # ---------------------------
    # 1) Query expansion
    # ---------------------------
    try:
        expansions = expand_query(query) if use_expansion else []
    except Exception as e:
        logger.exception("Query expansion failed; continuing without expansion: %s", e)
        expansions = []
    expanded_query = " ".join([query] + expansions) if expansions else query

    # ---------------------------
    # 2) Embed the (expanded) query
    # ---------------------------
    try:
        qvec = embed_texts([expanded_query])  # shape (1, dim), dtype float32 (normalized)
    except Exception as e:
        logger.exception("Embedding query failed: %s", e)
        return {"results": [], "suggestions": [], "summary": "", "explanations": [], "expanded_terms": expansions}

    if qvec.dtype != np.float32:
        qvec = qvec.astype("float32")

    # ---------------------------
    # 3) Load FAISS index + doc ids
    # ---------------------------
    index, doc_ids = get_index_and_doc_ids()
    if index is None or getattr(index, "ntotal", 0) == 0 or not doc_ids:
        # no data yet
        return {"results": [], "suggestions": [], "summary": "", "explanations": [], "expanded_terms": expansions}

    # ---------------------------
    # 4) Retrieve candidate docs (get extra for reranking)
    # ---------------------------
    fetch_n = max(k * 4, k + 10)  # retrieve a buffer of candidates for reranking
    try:
        distances, indices = index.search(qvec, fetch_n)
    except Exception as e:
        logger.exception("FAISS search failed: %s", e)
        return {"results": [], "suggestions": [], "summary": "", "explanations": [], "expanded_terms": expansions}

    raw_idxs = indices[0].tolist()
    raw_scores = distances[0].tolist()

    # Map FAISS positions to doc ids and prepare candidate list
    candidate_doc_ids: List[str] = []
    candidate_scores: List[float] = []
    for idx, sc in zip(raw_idxs, raw_scores):
        if idx == -1:
            continue
        try:
            did = doc_ids[idx]
        except Exception:
            # index out of bounds or inconsistent state
            continue
        candidate_doc_ids.append(did)
        candidate_scores.append(float(sc))

    if not candidate_doc_ids:
        return {"results": [], "suggestions": [], "summary": "", "explanations": [], "expanded_terms": expansions}

    # ---------------------------
    # 5) Load metadata for candidates
    # ---------------------------
    metadata_map = _fetch_metadata_for_ids(candidate_doc_ids)

    candidates: List[Dict[str, Any]] = []
    texts_for_fuzzy: List[str] = []
    for did, sc in zip(candidate_doc_ids, candidate_scores):
        meta = metadata_map.get(did, {"id": did, "title": "", "text": ""})
        cand = {
            "id": did,
            "title": meta.get("title", "") or "",
            "text": meta.get("text", "") or "",
            "score": sc,  # raw FAISS score (inner product)
        }
        candidates.append(cand)
        texts_for_fuzzy.append((cand["title"] + " " + cand["text"]).strip())

    # ---------------------------
    # 6) Fuzzy matching feature
    # ---------------------------
    fuzzy_scores = _compute_fuzzy_scores(query, texts_for_fuzzy)
    for cand, fs in zip(candidates, fuzzy_scores):
        cand["fuzzy_score"] = float(fs)

    # ---------------------------
    # 7) Re-rank candidates (optional)
    # ---------------------------
    try:
        if rerank:
            # rerank_with_cross_encoder returns candidates sorted by rerank_score
            candidates = rerank_with_cross_encoder(query, candidates, top_k=max(k * 2, len(candidates)))
    except Exception as e:
        logger.exception("Re-ranking failed; continuing with original ordering: %s", e)

    # Keep only top-k candidates
    candidates = candidates[:k]

    # ---------------------------
    # 8) Filter by minimum score threshold
    # ---------------------------
    def _get_effective_score(c: Dict[str, Any]) -> float:
        # prefer rerank_score if present, otherwise raw score
        return float(c.get("rerank_score", c.get("score", 0.0)))

    if min_score > 0.0:
        candidates = [c for c in candidates if _get_effective_score(c) >= min_score]

    # ---------------------------
    # 9) Generate "Did you mean?" suggestions (from sample of titles)
    # ---------------------------
    suggestions: List[Tuple[str, int]] = []
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT title FROM docs LIMIT 500")
        title_rows = cur.fetchall()
        conn.close()
        titles_sample = [r["title"] or "" for r in title_rows]
        suggestions = did_you_mean(query, titles_sample, limit=3)
    except Exception as e:
        logger.debug("Could not build suggestions from DB: %s", e)
        suggestions = []

    # ---------------------------
    # 10) Highlight matched terms in results
    # ---------------------------
    query_terms = [query] + expansions
    for c in candidates:
        try:
            c["highlighted_text"] = _highlight_text(c.get("text", ""), query_terms)
        except Exception:
            c["highlighted_text"] = c.get("text", "")

    # ---------------------------
    # 11) Optional clustering on final result set
    # ---------------------------
    if cluster and candidates:
        try:
            candidate_texts = [c["text"] for c in candidates]
            if len(candidate_texts) > 0:
                emb_vectors = embed_texts(candidate_texts)
                if emb_vectors is not None and emb_vectors.shape[0] > 1:
                    candidates = cluster_results_by_embeddings(
                        candidates, emb_vectors, n_clusters=min(4, len(candidates))
                    )
        except Exception as e:
            logger.debug("Clustering failed or skipped: %s", e)

    # ---------------------------
    # 12) Summarize top results (extractive)
    # ---------------------------
    try:
        summary = summarize_texts([c["text"] for c in candidates], n_sentences=3)
    except Exception as e:
        logger.debug("Summarization failed: %s", e)
        summary = ""

    # ---------------------------
    # 13) Build explanations (score list for plotting)
    # ---------------------------
    explanations = [_get_effective_score(c) for c in candidates]

    return {
        "results": candidates,
        "suggestions": suggestions,
        "summary": summary,
        "explanations": explanations,
        "expanded_terms": expansions,
    }
