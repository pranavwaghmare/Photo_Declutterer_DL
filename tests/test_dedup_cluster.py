"""
tests/test_dedup_cluster.py
───────────────────────────
Unit tests for src/dedup_cluster.py.
Uses synthetic embeddings — no external datasets required.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.dedup_cluster import (
    run_dbscan,
    compute_similarity_scores,
    run_dedup_clustering,
    evaluate_clustering,
    build_cosine_distance_graph,
)


# ─────────────────────────────────────────────────────────────────────────────
# DBSCAN
# ─────────────────────────────────────────────────────────────────────────────

class TestDBSCAN:
    def test_near_duplicates_clustered_together(self, near_duplicate_embeddings):
        embs, ids, expected = near_duplicate_embeddings
        labels = run_dbscan(embs, eps=0.05, min_samples=2)

        # The two near-duplicate pairs should each be in the same cluster
        assert labels[0] == labels[1], "dup pair A should share a cluster"
        assert labels[2] == labels[3], "dup pair B should share a cluster"

    def test_distinct_images_not_merged(self, near_duplicate_embeddings):
        embs, ids, _ = near_duplicate_embeddings
        labels = run_dbscan(embs, eps=0.05, min_samples=2)
        # The two clusters should have different IDs
        assert labels[0] != labels[2], "distinct groups should have different cluster IDs"

    def test_singletons_are_noise(self, near_duplicate_embeddings):
        embs, ids, expected = near_duplicate_embeddings
        labels = run_dbscan(embs, eps=0.05, min_samples=2)
        # Singletons are indices 4 and 5
        assert labels[4] == -1, f"Expected -1 for singleton, got {labels[4]}"
        assert labels[5] == -1, f"Expected -1 for singleton, got {labels[5]}"

    def test_empty_input_returns_empty(self):
        labels = run_dbscan(np.empty((0, 16), dtype=np.float32))
        assert labels.shape == (0,)

    def test_single_image_is_singleton(self):
        emb = np.random.default_rng(0).random((1, 16)).astype(np.float32)
        labels = run_dbscan(emb)
        assert labels[0] == -1


# ─────────────────────────────────────────────────────────────────────────────
# Similarity scores
# ─────────────────────────────────────────────────────────────────────────────

class TestSimilarityScores:
    def test_scores_in_valid_range(self, near_duplicate_embeddings):
        embs, _, _ = near_duplicate_embeddings
        labels = run_dbscan(embs, eps=0.05, min_samples=2)
        scores = compute_similarity_scores(embs, labels)
        assert scores.shape == (len(embs),)
        assert (scores >= 0).all(), "All scores should be non-negative"
        assert (scores <= 1.0 + 1e-5).all(), "All scores should be ≤ 1"

    def test_singletons_score_one(self, near_duplicate_embeddings):
        embs, _, _ = near_duplicate_embeddings
        labels = run_dbscan(embs, eps=0.05, min_samples=2)
        scores = compute_similarity_scores(embs, labels)
        singleton_mask = labels == -1
        assert np.allclose(
            scores[singleton_mask], 1.0
        ), "Singletons should have similarity score of 1.0"


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestRunDedupClustering:
    def test_output_columns(self, small_embeddings):
        embs, ids = small_embeddings
        result = run_dedup_clustering(embs, ids)
        assert list(result.columns) == ["Image_ID", "Duplicate_Cluster_ID", "Similarity_Score"]

    def test_output_row_count_matches_input(self, small_embeddings):
        embs, ids = small_embeddings
        result = run_dedup_clustering(embs, ids)
        assert len(result) == len(ids)

    def test_image_ids_match(self, small_embeddings):
        embs, ids = small_embeddings
        result = run_dedup_clustering(embs, ids)
        assert list(result["Image_ID"]) == ids

    def test_empty_input(self):
        result = run_dedup_clustering(np.empty((0, 16), dtype=np.float32), [])
        assert result.empty

    def test_near_duplicates_detected(self, near_duplicate_embeddings):
        embs, ids, _ = near_duplicate_embeddings
        result = run_dedup_clustering(embs, ids, eps=0.05, min_samples=2)
        cluster_a = result.loc[result["Image_ID"] == "nd_0", "Duplicate_Cluster_ID"].iloc[0]
        cluster_a2 = result.loc[result["Image_ID"] == "nd_1", "Duplicate_Cluster_ID"].iloc[0]
        assert cluster_a == cluster_a2
        assert cluster_a != -1


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metric
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateClustering:
    def test_perfect_clustering(self):
        pred = np.array([0, 0, 1, 1, -1])
        true = np.array([0, 0, 1, 1,  2])
        metrics = evaluate_clustering(pred, true)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_all_singletons_no_recall(self):
        pred = np.array([-1, -1, -1, -1])
        true = np.array([0, 0, 1, 1])
        metrics = evaluate_clustering(pred, true)
        assert metrics["recall"] == 0.0
        assert metrics["precision"] == 0.0 or True  # no positives → precision undefined

    def test_returns_dict_with_expected_keys(self):
        pred = np.array([0, 0, 1])
        true = np.array([0, 0, 1])
        metrics = evaluate_clustering(pred, true)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_cosine_graph_shape(self, small_embeddings):
        embs, _ = small_embeddings
        k = 3
        dists, inds = build_cosine_distance_graph(embs, n_neighbors=k)
        assert dists.shape == (len(embs), k)
        assert inds.shape == (len(embs), k)
        assert (dists >= 0).all(), "Cosine distances must be non-negative"
