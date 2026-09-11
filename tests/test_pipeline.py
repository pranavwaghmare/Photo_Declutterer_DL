"""
tests/test_pipeline.py
──────────────────────
End-to-end integration tests for src/pipeline.py.
Uses a small synthetic image set as fixtures — no downloads required.

Tests verify:
    1. ingest() produces the expected DataFrame schema.
    2. add_blur_scores() adds required columns.
    3. The recommend() function produces valid Keep/Delete labels.
    4. The full run_pipeline() produces a parquet with the correct schema.

Note: CLIP embedding extraction is mocked with random vectors to keep
tests fast and dependency-free (no network access required).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.ingest import ingest, scan_folder
from src.blur_quality import add_blur_scores
from src.recommend import recommend
from src.pipeline import SCHEMA_COLUMNS, run_pipeline, load_config


# ─────────────────────────────────────────────────────────────────────────────
# Ingest
# ─────────────────────────────────────────────────────────────────────────────

class TestIngest:
    def test_scan_folder_finds_images(self, sample_image_dir):
        paths = scan_folder(sample_image_dir)
        assert len(paths) >= 8, f"Expected ≥8 images, found {len(paths)}"

    def test_ingest_schema(self, sample_image_dir, tmp_path):
        df = ingest(sample_image_dir, output_path=tmp_path / "meta.parquet")
        expected_cols = {
            "Image_ID", "File_Path", "Capture_Timestamp", "Width", "Height", "File_Size_Bytes"
        }
        assert expected_cols.issubset(set(df.columns))

    def test_ingest_no_duplicate_ids(self, sample_image_dir, tmp_path):
        df = ingest(sample_image_dir)
        assert df["Image_ID"].nunique() == len(df), "All Image_IDs should be unique"

    def test_ingest_nonexistent_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ingest(tmp_path / "does_not_exist")

    def test_parquet_saved(self, sample_image_dir, tmp_path):
        out = tmp_path / "meta.parquet"
        ingest(sample_image_dir, output_path=out)
        assert out.exists()
        loaded = pd.read_parquet(out)
        assert len(loaded) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Blur quality
# ─────────────────────────────────────────────────────────────────────────────

class TestBlurIntegration:
    def test_laplacian_only_adds_columns(self, sample_image_dir):
        df = ingest(sample_image_dir)
        result = add_blur_scores(df, model=None, laplacian_threshold=100.0)
        assert "Laplacian_Variance" in result.columns
        assert "Blur_Quality_Label" in result.columns

    def test_sharp_images_labeled_correctly(self, sample_image_dir):
        df = ingest(sample_image_dir)
        result = add_blur_scores(df, model=None, laplacian_threshold=1.0)
        # With threshold=1 almost all real images should be "sharp"
        assert (result["Blur_Quality_Label"] == "sharp").any()

    def test_blurry_images_labeled_correctly(self, sample_image_dir):
        df = ingest(sample_image_dir)
        result = add_blur_scores(df, model=None, laplacian_threshold=1e9)
        # With impossibly high threshold, all images should be "blurry"
        assert (result["Blur_Quality_Label"] == "blurry").all()


# ─────────────────────────────────────────────────────────────────────────────
# Recommend
# ─────────────────────────────────────────────────────────────────────────────

class TestRecommend:
    def _make_df(self, n: int = 6) -> pd.DataFrame:
        rng = np.random.default_rng(99)
        cluster_ids = [0, 0, 1, 1, -1, -1][:n]
        return pd.DataFrame(
            {
                "Image_ID": [f"img_{i}" for i in range(n)],
                "Duplicate_Cluster_ID": cluster_ids,
                "Similarity_Score": rng.random(n).tolist(),
                "Laplacian_Variance": [200.0, 50.0, 150.0, 30.0, 120.0, 10.0][:n],
                "Width": [1920] * n,
                "Height": [1080] * n,
            }
        )

    def test_output_has_keep_recommendation_column(self):
        df = self._make_df()
        result = recommend(df)
        assert "Keep_Recommendation" in result.columns

    def test_values_are_keep_or_delete(self):
        df = self._make_df()
        result = recommend(df)
        assert set(result["Keep_Recommendation"].unique()).issubset({"Keep", "Delete"})

    def test_exactly_one_keep_per_duplicate_cluster(self):
        df = self._make_df()
        result = recommend(df)
        clustered = result[result["Duplicate_Cluster_ID"] != -1]
        for cluster_id, group in clustered.groupby("Duplicate_Cluster_ID"):
            n_keep = (group["Keep_Recommendation"] == "Keep").sum()
            assert n_keep == 1, f"Cluster {cluster_id}: expected 1 Keep, got {n_keep}"

    def test_high_quality_singleton_is_kept(self):
        df = self._make_df()
        result = recommend(df, quality_delete_threshold=50.0)
        # img_4 has Laplacian_Variance=120 → above threshold → Keep
        assert result.loc[result["Image_ID"] == "img_4", "Keep_Recommendation"].iloc[0] == "Keep"

    def test_low_quality_singleton_is_deleted(self):
        df = self._make_df()
        result = recommend(df, quality_delete_threshold=50.0)
        # img_5 has Laplacian_Variance=10 → below threshold → Delete
        assert result.loc[result["Image_ID"] == "img_5", "Keep_Recommendation"].iloc[0] == "Delete"


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline (with mocked CLIP)
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndPipeline:
    """
    Runs the full pipeline on the synthetic image fixture but mocks out
    CLIP (load_clip_model + extract_embeddings) so tests are fast and
    require no network access.
    """

    def _mock_clip(self, n_images: int, d: int = 32):
        """Return mock (model, preprocess, device) and embeddings."""
        mock_model = MagicMock()
        mock_model.encode_text = MagicMock(
            side_effect=lambda tokens: torch.randn(len(tokens), d)
        )
        mock_preprocess = MagicMock(side_effect=lambda img: torch.zeros(3, 224, 224))
        mock_device = torch.device("cpu")

        rng = np.random.default_rng(7)
        raw = rng.random((n_images, d)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        embs = raw / norms
        return mock_model, mock_preprocess, mock_device, embs

    def test_full_pipeline_produces_schema(self, sample_image_dir, tmp_path):
        from src.ingest import ingest as _ingest  # noqa: PLC0415

        # Pre-run ingest to know the real Image_IDs the pipeline will use
        real_df = _ingest(sample_image_dir)
        real_ids = real_df["Image_ID"].tolist()
        n = len(real_ids)
        mock_model, mock_preprocess, mock_device, embs = self._mock_clip(n)
        # Trim / pad embeddings to exactly n rows
        embs = embs[:n]

        with (
            patch("src.pipeline.load_clip_model", return_value=(mock_model, mock_preprocess, mock_device)),
            patch(
                "src.pipeline.extract_embeddings",
                return_value=(embs, real_ids),
            ),
        ):
            config = {
                "paths": {"reports_dir": str(tmp_path / "reports")},
                "embeddings": {"clip_model": "ViT-B-32", "clip_pretrained": "openai", "device": "cpu", "batch_size": 4},
                "blur_quality": {"laplacian_threshold": 100.0, "model": "laplacian_only"},
                "dedup_cluster": {"eps": 0.3, "min_samples": 2, "ann_neighbors": 5},
                "content_cluster": {"n_clusters": 3, "max_k": 5, "random_state": 42},
                "recommend": {"quality_delete_threshold": 10.0},
            }
            out_parquet = tmp_path / "results.parquet"
            result = run_pipeline(
                input_dir=sample_image_dir,
                output_path=out_parquet,
                config=config,
            )

        assert not result.empty, "Pipeline should produce a non-empty DataFrame"
        for col in SCHEMA_COLUMNS:
            assert col in result.columns, f"Missing schema column: {col}"

    def test_output_parquet_saved(self, sample_image_dir, tmp_path):
        from src.ingest import ingest as _ingest  # noqa: PLC0415

        real_df = _ingest(sample_image_dir)
        real_ids = real_df["Image_ID"].tolist()
        n = len(real_ids)
        mock_model, mock_preprocess, mock_device, embs = self._mock_clip(n)
        embs = embs[:n]

        with (
            patch("src.pipeline.load_clip_model", return_value=(mock_model, mock_preprocess, mock_device)),
            patch(
                "src.pipeline.extract_embeddings",
                return_value=(embs, real_ids),
            ),
        ):
            config = {
                "paths": {"reports_dir": str(tmp_path / "reports")},
                "embeddings": {"device": "cpu", "batch_size": 4},
                "blur_quality": {"laplacian_threshold": 100.0, "model": "laplacian_only"},
                "dedup_cluster": {"eps": 0.3, "min_samples": 2, "ann_neighbors": 5},
                "content_cluster": {"n_clusters": 2, "max_k": 5, "random_state": 42},
                "recommend": {"quality_delete_threshold": 10.0},
            }
            out_parquet = tmp_path / "results.parquet"
            run_pipeline(sample_image_dir, out_parquet, config)

        assert out_parquet.exists()
        df = pd.read_parquet(out_parquet)
        assert len(df) > 0

    def test_load_config_missing_returns_empty_dict(self, tmp_path):
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert isinstance(cfg, dict)
        assert len(cfg) == 0
