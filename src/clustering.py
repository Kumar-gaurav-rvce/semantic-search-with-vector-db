# src/clustering.py
"""
Clustering utilities for semantic search results.

We use k-means (via scikit-learn) to assign cluster labels to embeddings.
These cluster labels can be displayed in the Streamlit app to group similar
search results together.

Functions:
    cluster_embeddings: Cluster embeddings into groups and return labels.
    cluster_results_by_embeddings: Attach cluster labels to result dicts.
"""

from typing import List, Dict
import numpy as np
from sklearn.cluster import KMeans


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int = 5) -> List[int]:
    """
    Cluster embeddings using k-means and return the cluster label for each vector.

    Args:
        embeddings (np.ndarray): 2D array of shape (n_samples, dim).
        n_clusters (int): Number of clusters to create (default = 5).

    Returns:
        List[int]: Cluster labels, one per input embedding.

    Notes:
        - If fewer samples than requested clusters, number of clusters is reduced.
        - Uses fixed random_state for reproducibility.
    """
    if embeddings is None or embeddings.shape[0] == 0:
        return []

    # Use fewer clusters if we have fewer points
    k = min(n_clusters, embeddings.shape[0])

    # Initialize KMeans with multiple runs (n_init) for stability
    km = KMeans(n_clusters=k, random_state=42, n_init=10)

    # Fit + predict cluster labels
    labels = km.fit_predict(embeddings)

    return labels.tolist()


def cluster_results_by_embeddings(
    results: List[Dict], embeddings: np.ndarray, n_clusters: int = 5
) -> List[Dict]:
    """
    Attach a cluster label to each result dict, based on its embedding.

    Args:
        results (List[Dict]): Search result objects (with metadata).
        embeddings (np.ndarray): 2D array of embeddings, same order as results.
        n_clusters (int): Desired number of clusters.

    Returns:
        List[Dict]: Same results with an extra key `"cluster"`.

    Example:
        results = [{"id": 1, "text": "Airport run"}, {"id": 2, "text": "Heavy traffic"}]
        labels = cluster_results_by_embeddings(results, embeddings, n_clusters=2)
        # -> results[0]["cluster"] = 0, results[1]["cluster"] = 1
    """
    labels = cluster_embeddings(embeddings, n_clusters=n_clusters)

    # Attach labels back into the results
    for r, lab in zip(results, labels):
        r["cluster"] = int(lab)

    return results
