"""
recommend.py
────────────
Per-cluster "best photo" selection → Keep_Recommendation column.

Algorithm
---------
Within each Duplicate_Cluster_ID group (of near-duplicates):
    1. Score each image on a composite metric:
           composite = w_lap * norm(Laplacian_Variance)
                     + w_res * norm(Width * Height)
                     + w_sim * Similarity_Score
    2. Mark the top-scored image as ``Keep``; all others as ``Delete``.

Singleton images (Duplicate_Cluster_ID == -1):
    - Laplacian_Variance ≥ quality_delete_threshold → ``Keep``
    - Laplacian_Variance <  quality_delete_threshold → ``Delete``
      (flagged, never auto-removed from disk)

Public API
----------
recommend(df, quality_delete_threshold, weights) → df with Keep_Recommendation
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default composite score weights (must sum to 1.0)
DEFAULT_WEIGHTS: Tuple[float, float, float] = (0.5, 0.3, 0.2)
# (laplacian_weight, resolution_weight, similarity_weight)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _minmax_norm(series: pd.Series) -> pd.Series:
    """Min-max normalise a Series to [0, 1]; returns 0.5 if constant."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def _composite_score(
    df: pd.DataFrame,
    w_lap: float,
    w_res: float,
    w_sim: float,
) -> pd.Series:
    """Compute a composite quality score for every row in *df*."""
    lap_norm = _minmax_norm(df["Laplacian_Variance"].fillna(0.0))

    if "Width" in df.columns and "Height" in df.columns:
        resolution = (df["Width"] * df["Height"]).fillna(0)
    else:
        resolution = pd.Series(1, index=df.index)
    res_norm = _minmax_norm(resolution.astype(float))

    sim = df["Similarity_Score"].fillna(1.0)

    return w_lap * lap_norm + w_res * res_norm + w_sim * sim


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def recommend(
    df: pd.DataFrame,
    quality_delete_threshold: float = 50.0,
    weights: Tuple[float, float, float] = DEFAULT_WEIGHTS,
) -> pd.DataFrame:
    """
    Assign ``Keep_Recommendation`` to every row in *df*.

    Parameters
    ----------
    df:
        Must contain:
            Image_ID, Duplicate_Cluster_ID, Similarity_Score,
            Laplacian_Variance
        Optionally: Width, Height
    quality_delete_threshold:
        Laplacian variance below which a singleton image is recommended
        for deletion (flagged — never auto-deleted).
    weights:
        (w_laplacian, w_resolution, w_similarity) composite score weights.

    Returns
    -------
    Copy of *df* with ``Keep_Recommendation`` column added:
        ``"Keep"`` or ``"Delete"``.
    """
    df = df.copy()
    w_lap, w_res, w_sim = weights
    df["_composite"] = _composite_score(df, w_lap, w_res, w_sim)

    recommendations = pd.Series("Keep", index=df.index, dtype=str)

    # ── Duplicate clusters ─────────────────────────────────────────────────
    clustered_mask = df["Duplicate_Cluster_ID"] != -1
    clustered_df = df[clustered_mask]

    for cluster_id, group in clustered_df.groupby("Duplicate_Cluster_ID"):
        best_idx = group["_composite"].idxmax()
        delete_idxs = group.index.difference([best_idx])
        recommendations[delete_idxs] = "Delete"
        # Best image in cluster always kept regardless of quality threshold
        recommendations[best_idx] = "Keep"

    # ── Singletons ─────────────────────────────────────────────────────────
    singleton_mask = df["Duplicate_Cluster_ID"] == -1
    below_threshold = singleton_mask & (df["Laplacian_Variance"] < quality_delete_threshold)
    recommendations[below_threshold] = "Delete"

    df["Keep_Recommendation"] = recommendations
    df = df.drop(columns=["_composite"])

    n_keep = (df["Keep_Recommendation"] == "Keep").sum()
    n_delete = (df["Keep_Recommendation"] == "Delete").sum()
    logger.info(
        "Recommendations: %d Keep, %d Delete (%.1f%% reduction).",
        n_keep,
        n_delete,
        100.0 * n_delete / max(1, len(df)),
    )
    return df
