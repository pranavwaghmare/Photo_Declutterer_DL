"""
tests/test_content_cluster.py
─────────────────────────────
Unit tests for src/content_cluster.py.
Uses synthetic embeddings — no external datasets required.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock
import torch

from src.content_cluster import (
    select_k,
    run_content_clustering,
    visualise_clusters,
    _zero_shot_label_clusters,
)


class TestContentCluster:
    def test_select_k_returns_valid_k(self, small_embeddings):
        embs, _ = small_embeddings
        k = select_k(embs, max_k=5)
        assert 1 <= k <= min(5, len(embs) - 1)

    def test_select_k_too_few_samples(self):
        embs = np.random.randn(2, 16).astype(np.float32)
        k = select_k(embs, max_k=5)
        assert k == 1

    def test_run_content_clustering_output_schema(self, small_embeddings):
        embs, ids = small_embeddings
        df = run_content_clustering(embs, ids, n_clusters=3)
        assert list(df.columns) == ["Image_ID", "Content_Category"]
        assert len(df) == len(ids)
        assert list(df["Image_ID"]) == ids
        assert all(isinstance(c, str) for c in df["Content_Category"])

    def test_run_content_clustering_empty(self):
        df = run_content_clustering(np.empty((0, 16), dtype=np.float32), [])
        assert df.empty
        assert list(df.columns) == ["Image_ID", "Content_Category"]

    def test_zero_shot_labelling_mock(self):
        centroids = np.random.randn(2, 32).astype(np.float32)
        candidate_labels = ["nature and landscapes", "food and drinks"]
        mock_model = MagicMock()
        mock_model.encode_text = MagicMock(
            side_effect=lambda tokens: torch.randn(len(tokens), 32)
        )
        labels = _zero_shot_label_clusters(
            centroids,
            candidate_labels,
            clip_model=mock_model,
            clip_tokenizer=None,
            device=torch.device("cpu"),
        )
        assert isinstance(labels, dict)
        assert len(labels) == 2
        for k, v in labels.items():
            assert v in candidate_labels

    def test_visualise_clusters_saves_png(self, small_embeddings, tmp_path):
        embs, ids = small_embeddings
        labels = ["landscape" if i % 2 == 0 else "portrait" for i in range(len(ids))]
        out_png = tmp_path / "clusters.png"
        saved = visualise_clusters(embs, labels, output_path=out_png, method="tsne")
        assert saved.exists()
        assert saved.stat().st_size > 0
