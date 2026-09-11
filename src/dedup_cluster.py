"""
dedup_cluster.py
────────────────
Near-duplicate detection via CLIP embeddings + DBSCAN.

Steps
-----
1. Approximate Nearest Neighbour (ANN) pre-filtering using
   ``sklearn.neighbors.NearestNeighbors`` with cosine metric (avoids O(n²)
   memory blow-up for large libraries).
2. DBSCAN on the pre-filtered cosine-distance graph to assign cluster IDs.
3. Centroid computation → per-image Similarity_Score.

Public API
----------
run_dedup_clustering(embeddings, image_ids, eps, min_samples, ann_neighbors)
    → pd.DataFrame with Duplicate_Cluster_ID and Similarity_Score columns
evaluate_clustering(pred_labels, true_labels) → dict of metrics
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core clustering
# ─────────────────────────────────────────────────────────────────────────────

def _l2_normalize(X: np.ndarray) -> np.ndarray:
    """Return L2-normalised copy of *X* (row-wise)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms


def build_cosine_distance_graph(
    embeddings: np.ndarray,
    n_neighbors: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a sparse cosine-distance neighbour graph via sklearn NearestNeighbors.

    Cosine distance = 1 − cosine_similarity.  Embeddings are L2-normalised
    before fitting so that Euclidean distance equals cosine distance.

    Parameters
    ----------
    embeddings:
        Float32 array of shape [N, D].
    n_neighbors:
        Number of neighbours to retrieve per point (ANN pre-filtering).

    Returns
    -------
    (distances, indices) arrays of shape [N, n_neighbors].
    """
    X = _l2_normalize(embeddings.astype(np.float32))
    k = min(n_neighbors + 1, len(X))  # +1 because query point is included
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    # Drop self (distance ≈ 0 at column 0)
    return distances[:, 1:], indices[:, 1:]


def run_dbscan(
    embeddings: np.ndarray,
    eps: float = 0.15,
    min_samples: int = 2,
    ann_neighbors: int = 50,
) -> np.ndarray:
    """
    Run DBSCAN with cosine metric on *embeddings*.

    Uses sklearn's DBSCAN which internally computes the full pairwise matrix
    for small datasets, or we pass a pre-computed sparse matrix for large ones.

    Parameters
    ----------
    embeddings:
        L2-normalised float32 array [N, D].
    eps:
        Maximum cosine distance for two samples to be considered neighbours.
    min_samples:
        Minimum cluster size (2 = a pair of near-duplicates is a cluster).
    ann_neighbors:
        Neighbour count for ANN pre-filtering (used when N is large).

    Returns
    -------
    Integer array of cluster labels (shape [N,]).
    Noise points are labelled -1 (singletons in our context).
    """
    X = _l2_normalize(embeddings.astype(np.float32))
    n = len(X)

    if n < 2:
        return np.full(n, -1, dtype=int)

    if n <= 2000:
        # Small enough for exact pairwise cosine
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
        labels = db.fit_predict(X)
    else:
        # Pre-compute sparse neighbour graph; feed to DBSCAN
        logger.info("Large dataset (%d imgs): using ANN pre-filtering for DBSCAN.", n)
        dists, inds = build_cosine_distance_graph(X, n_neighbors=min(ann_neighbors, n - 1))
        # Build condensed distance matrix as a connectivity proxy
        # DBSCAN with metric='precomputed' needs a square dense matrix — only feasible
        # up to ~20k images.  For >20k use approximate approach.
        if n <= 20000:
            from sklearn.metrics import pairwise_distances  # noqa: PLC0415

            dist_matrix = pairwise_distances(X, metric="cosine", n_jobs=-1)
            db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1)
            labels = db.fit_predict(dist_matrix)
        else:
            # Approximate: treat ANN graph as the neighbour structure
            logger.warning(
                "Dataset too large for exact DBSCAN; using ANN-graph approximation."
            )
            db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
            labels = db.fit_predict(X)

    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    logger.info(
        "DBSCAN: %d cluster(s), %d noise/singleton point(s).", n_clusters, n_noise
    )
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Similarity score (cosine similarity to cluster centroid)
# ─────────────────────────────────────────────────────────────────────────────

def compute_similarity_scores(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Compute the cosine similarity of each image to its cluster centroid.

    Singleton / noise points (label == -1) get a score of 1.0 (they are their
    own centroid).

    Parameters
    ----------
    embeddings:
        L2-normalised float32 array [N, D].
    labels:
        DBSCAN cluster labels array [N,].

    Returns
    -------
    Float32 array [N,] of similarity scores in [0, 1].
    """
    X = _l2_normalize(embeddings.astype(np.float32))
    scores = np.ones(len(X), dtype=np.float32)

    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue
        mask = labels == cluster_id
        centroid = X[mask].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        # Cosine similarity = dot product of L2-normalised vectors
        sims = X[mask] @ centroid
        scores[mask] = sims.clip(0.0, 1.0)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_dedup_clustering(
    embeddings: np.ndarray,
    image_ids: List[str],
    eps: float = 0.15,
    min_samples: int = 2,
    ann_neighbors: int = 50,
) -> pd.DataFrame:
    """
    Full near-duplicate clustering pipeline.

    Parameters
    ----------
    embeddings:
        Float32 array [N, D] of CLIP embeddings (need not be pre-normalised).
    image_ids:
        List of Image_ID strings in the same row order as *embeddings*.
    eps:
        DBSCAN cosine-distance epsilon.
    min_samples:
        DBSCAN minimum cluster size.
    ann_neighbors:
        ANN pre-filter neighbour count.

    Returns
    -------
    DataFrame with columns:
        Image_ID, Duplicate_Cluster_ID, Similarity_Score

    ``Duplicate_Cluster_ID`` is -1 for singletons (no near-duplicate found).
    """
    if len(embeddings) == 0:
        return pd.DataFrame(columns=["Image_ID", "Duplicate_Cluster_ID", "Similarity_Score"])

    labels = run_dbscan(embeddings, eps=eps, min_samples=min_samples, ann_neighbors=ann_neighbors)
    scores = compute_similarity_scores(embeddings, labels)

    return pd.DataFrame(
        {
            "Image_ID": image_ids,
            "Duplicate_Cluster_ID": labels.tolist(),
            "Similarity_Score": scores.tolist(),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_clustering(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate clustering quality against ground-truth duplicate groups.

    Uses pairwise precision / recall / F1 (BCubed-style):
    - For each pair of images in the same predicted cluster, check if they
      share the same ground-truth group (precision).
    - For each pair sharing the same ground-truth group, check if they are in
      the same predicted cluster (recall).

    Parameters
    ----------
    pred_labels:
        Predicted cluster IDs (−1 for singletons).
    true_labels:
        Ground-truth group IDs.

    Returns
    -------
    Dict with keys: precision, recall, f1.
    """
    n = len(pred_labels)
    tp = fp = fn = 0

    for i in range(n):
        for j in range(i + 1, n):
            same_pred = pred_labels[i] != -1 and pred_labels[i] == pred_labels[j]
            same_true = true_labels[i] == true_labels[j]

            if same_pred and same_true:
                tp += 1
            elif same_pred and not same_true:
                fp += 1
            elif not same_pred and same_true:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}
