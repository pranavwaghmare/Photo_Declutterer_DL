"""
content_cluster.py
──────────────────
K-Means content-category clustering on CLIP embeddings.

Steps
-----
1. Optionally auto-select K via elbow / silhouette analysis.
2. Run K-Means on L2-normalised embeddings.
3. (Optional) Auto-label each cluster via CLIP zero-shot classification.
4. Produce a 2-D UMAP / t-SNE visualisation saved to the reports directory.

Public API
----------
select_k(embeddings, max_k, random_state)   → int
run_content_clustering(embeddings, image_ids, n_clusters, ...)
    → pd.DataFrame  with Content_Category column
visualise_clusters(embeddings, labels, ...)  → saved PNG path
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

# Default candidate labels for zero-shot cluster naming
DEFAULT_LABELS: List[str] = [
    "people and portraits",
    "food and drinks",
    "documents and text",
    "nature and landscapes",
    "screenshots and UI",
    "animals and pets",
    "architecture and buildings",
    "events and celebrations",
    "travel and tourism",
    "abstract and art",
]


# ─────────────────────────────────────────────────────────────────────────────
# K selection
# ─────────────────────────────────────────────────────────────────────────────

def select_k(
    embeddings: np.ndarray,
    max_k: int = 20,
    random_state: int = 42,
) -> int:
    """
    Choose the number of clusters K via silhouette analysis.

    Evaluates K in range [2, min(max_k, n-1)] and returns the K with the
    highest average silhouette score.

    Parameters
    ----------
    embeddings:
        Float32 array [N, D].  Need not be pre-normalised.
    max_k:
        Upper bound on K to try.
    random_state:
        Random seed for K-Means.

    Returns
    -------
    Optimal K as an integer.
    """
    X = normalize(embeddings.astype(np.float32))
    n = len(X)
    upper = min(max_k, n - 1)

    if upper < 2:
        logger.warning("Too few samples (%d) to run silhouette analysis; using K=1.", n)
        return 1

    best_k, best_score = 2, -1.0
    for k in range(2, upper + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels, metric="cosine", sample_size=min(5000, n))
        logger.debug("  K=%d  silhouette=%.4f", k, score)
        if score > best_score:
            best_score = score
            best_k = k

    logger.info("Selected K=%d (silhouette=%.4f).", best_k, best_score)
    return best_k


# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot cluster labelling via CLIP
# ─────────────────────────────────────────────────────────────────────────────

def _zero_shot_label_clusters(
    cluster_centroids: np.ndarray,
    candidate_labels: List[str],
    clip_model,
    clip_tokenizer,
    device,
) -> Dict[int, str]:
    """
    For each cluster centroid, find the candidate label whose CLIP text
    embedding has the highest cosine similarity.

    Returns dict mapping cluster_id → label string.
    """
    if clip_model is None or not candidate_labels or len(cluster_centroids) == 0:
        return {}

    import torch  # noqa: PLC0415

    try:
        import open_clip  # noqa: PLC0415

        tokens = open_clip.tokenize(candidate_labels).to(device)
        with torch.no_grad():
            text_feats = clip_model.encode_text(tokens)
            if hasattr(text_feats, "norm"):
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            if hasattr(text_feats, "cpu"):
                text_feats = text_feats.cpu().float().numpy()
            text_feats = np.asarray(text_feats, dtype=np.float32)

        if text_feats.ndim != 2 or text_feats.shape[0] != len(candidate_labels):
            return {}

        centroids = normalize(np.asarray(cluster_centroids, dtype=np.float32))
        if centroids.ndim == 1:
            centroids = centroids.reshape(1, -1)
        sims = centroids @ text_feats.T  # [K, L]
        best_label_ids = np.asarray(sims.argmax(axis=1)).flatten()
        return {i: str(candidate_labels[int(lid)]) for i, lid in enumerate(best_label_ids)}
    except Exception as exc:
        logger.warning("Zero-shot labelling failed: %s", exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Main clustering routine
# ─────────────────────────────────────────────────────────────────────────────

def run_content_clustering(
    embeddings: np.ndarray,
    image_ids: List[str],
    n_clusters: Optional[int] = 10,
    max_k: int = 20,
    random_state: int = 42,
    candidate_labels: Optional[List[str]] = None,
    clip_model=None,
    clip_tokenizer=None,
    clip_device=None,
) -> pd.DataFrame:
    """
    Cluster images by content using K-Means on CLIP embeddings.

    Parameters
    ----------
    embeddings:
        Float32 array [N, D].
    image_ids:
        List of Image_ID strings matching row order of *embeddings*.
    n_clusters:
        Fixed K; pass ``None`` to auto-select via silhouette analysis.
    max_k:
        Upper bound when auto-selecting K.
    random_state:
        Random seed.
    candidate_labels:
        Optional list of text labels for zero-shot cluster naming.
        If None, clusters are named "cluster_0", "cluster_1", …
    clip_model / clip_tokenizer / clip_device:
        Loaded CLIP model for zero-shot labelling; ignored if *candidate_labels*
        is None.

    Returns
    -------
    DataFrame with columns: Image_ID, Content_Category
    """
    if len(embeddings) == 0:
        return pd.DataFrame(columns=["Image_ID", "Content_Category"])

    X = normalize(embeddings.astype(np.float32))

    if n_clusters is None:
        n_clusters = select_k(X, max_k=max_k, random_state=random_state)

    n_clusters = min(n_clusters, len(X))
    logger.info("Running K-Means with K=%d on %d embeddings …", n_clusters, len(X))

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    raw_labels = km.fit_predict(X)

    # ── Optional zero-shot cluster naming ───────────────────────────────────
    centroids = km.cluster_centers_  # [K, D]
    label_map: Dict[int, str] = {}

    if candidate_labels and clip_model is not None:
        label_map = _zero_shot_label_clusters(
            centroids, candidate_labels, clip_model, clip_tokenizer, clip_device
        )

    if label_map:
        content_categories = [label_map.get(int(lbl), f"cluster_{lbl}") for lbl in raw_labels]
    else:
        content_categories = [f"cluster_{lbl}" for lbl in raw_labels]

    return pd.DataFrame({"Image_ID": image_ids, "Content_Category": content_categories})


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualise_clusters(
    embeddings: np.ndarray,
    labels: List[str],
    output_path: str | Path,
    method: str = "umap",
    random_state: int = 42,
    title: str = "Content Cluster Visualisation",
) -> Path:
    """
    Reduce embeddings to 2-D and save a scatter plot coloured by cluster label.

    Parameters
    ----------
    embeddings:
        Float32 array [N, D].
    labels:
        List of string category labels per image.
    output_path:
        Where to save the PNG.
    method:
        ``"umap"`` (preferred) or ``"tsne"``.
    random_state:
        Seed for reproducibility.
    title:
        Plot title.

    Returns
    -------
    Resolved :class:`pathlib.Path` to the saved PNG.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if len(embeddings) == 0:
        logger.warning("No embeddings to visualise.")
        return out

    X = normalize(np.asarray(embeddings, dtype=np.float32))
    n_samples = len(X)

    if n_samples < 2:
        coords = np.zeros((n_samples, 2), dtype=np.float32)
    else:
        coords = None
        if method == "umap":
            try:
                import umap  # noqa: PLC0415

                n_neighbors = min(15, max(2, n_samples - 1))
                reducer = umap.UMAP(
                    n_neighbors=n_neighbors,
                    n_components=2,
                    random_state=random_state,
                    verbose=False,
                )
                coords = reducer.fit_transform(X)
            except Exception as exc:
                logger.warning("UMAP failed (%s); falling back to t-SNE.", exc)
                method = "tsne"

        if coords is None and method == "tsne":
            try:
                from sklearn.manifold import TSNE  # noqa: PLC0415

                perplexity = min(30.0, max(1.0, (n_samples - 1.0) / 3.0))
                reducer = TSNE(
                    n_components=2,
                    random_state=random_state,
                    perplexity=perplexity,
                )
                coords = reducer.fit_transform(X)
            except Exception as exc:
                logger.warning("t-SNE failed (%s); using PCA fallback.", exc)
                from sklearn.decomposition import PCA  # noqa: PLC0415

                pca = PCA(n_components=min(2, X.shape[1], n_samples), random_state=random_state)
                coords_pca = pca.fit_transform(X)
                if coords_pca.shape[1] < 2:
                    coords = np.pad(coords_pca, ((0, 0), (0, 2 - coords_pca.shape[1])))
                else:
                    coords = coords_pca

    # Assign colour per unique label
    unique_labels = sorted(set(labels)) if labels else ["cluster"]
    n_labels = max(1, len(unique_labels))
    cmap = plt.colormaps.get_cmap("tab20").resampled(n_labels)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    colours = [cmap(label_to_idx.get(l, 0)) for l in labels] if labels else ["blue"] * n_samples

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], c=colours, s=20, alpha=0.7)

    # Legend
    handles = [
        plt.Line2D(
            [0], [0],
            marker="o", color="w",
            markerfacecolor=cmap(i),
            markersize=8,
            label=lbl,
        )
        for i, lbl in enumerate(unique_labels)
    ]
    ax.legend(handles=handles, loc="best", fontsize=8, ncol=max(1, min(3, len(unique_labels) // 5)))
    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Cluster visualisation saved to %s.", out)
    return out
