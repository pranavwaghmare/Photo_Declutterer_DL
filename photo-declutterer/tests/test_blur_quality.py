"""
tests/test_blur_quality.py
──────────────────────────
Unit tests for src/blur_quality.py.
All tests use synthetic images — no external datasets required.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.blur_quality import (
    add_blur_scores,
    score_laplacian,
    score_laplacian_batch,
    score_batch_model,
)
from src.models.cnn_blur import BlurCNN, build_blur_cnn
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Laplacian variance
# ─────────────────────────────────────────────────────────────────────────────

class TestLaplacianVariance:
    def test_sharp_image_high_variance(self, sample_image_dir):
        """A checkerboard (high-frequency) image should score above 100."""
        sharp_path = next(sample_image_dir.glob("sharp_*.jpg"))
        score = score_laplacian(str(sharp_path))
        assert score > 100.0, f"Expected score > 100, got {score}"

    def test_blurry_image_low_variance(self, sample_image_dir):
        """A uniform-grey (low-frequency) image should score near 0."""
        blurry_path = next(sample_image_dir.glob("blurry_*.jpg"))
        score = score_laplacian(str(blurry_path))
        assert score < 100.0, f"Expected score < 100, got {score}"

    def test_missing_file_returns_zero(self, tmp_path):
        """Non-existent file should return 0.0 without raising."""
        score = score_laplacian(str(tmp_path / "ghost.jpg"))
        assert score == 0.0

    def test_batch_returns_dict_with_all_ids(self, sample_image_dir):
        paths = list(sample_image_dir.glob("*.jpg"))
        ids = [p.stem for p in paths]
        fps = [str(p) for p in paths]
        result = score_laplacian_batch(ids, fps)
        assert set(result.keys()) == set(ids)
        assert all(isinstance(v, float) for v in result.values())


# ─────────────────────────────────────────────────────────────────────────────
# CNN model
# ─────────────────────────────────────────────────────────────────────────────

class TestBlurCNN:
    def test_output_shape(self):
        """BlurCNN should output logits of shape [B, 2]."""
        model = BlurCNN()
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 2), f"Expected (4, 2), got {out.shape}"

    def test_predict_proba_sums_to_one(self):
        """Softmax probabilities should sum to ~1.0."""
        model = BlurCNN()
        x = torch.randn(3, 3, 224, 224)
        probs = model.predict_proba(x)
        row_sums = probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(3), atol=1e-5)

    def test_build_blur_cnn_no_checkpoint(self):
        """build_blur_cnn(None) should return an eval-mode model."""
        model = build_blur_cnn(checkpoint=None)
        assert not model.training

    def test_build_blur_cnn_with_checkpoint(self, tmp_path):
        """build_blur_cnn should load weights from a saved state dict."""
        model = BlurCNN()
        ckpt = tmp_path / "blur_cnn.pth"
        torch.save(model.state_dict(), ckpt)
        loaded = build_blur_cnn(checkpoint=str(ckpt))
        assert not loaded.training
        # Verify weights match
        for (k1, v1), (k2, v2) in zip(
            model.state_dict().items(), loaded.state_dict().items()
        ):
            assert torch.allclose(v1, v2)


# ─────────────────────────────────────────────────────────────────────────────
# Model inference on synthetic images
# ─────────────────────────────────────────────────────────────────────────────

class TestModelInference:
    def test_score_batch_model_returns_dict(self, sample_image_dir):
        """score_batch_model should return a dict with the right keys."""
        paths = list(sample_image_dir.glob("*.jpg"))[:4]
        ids = [p.stem for p in paths]
        fps = [str(p) for p in paths]

        model = BlurCNN()
        result = score_batch_model(model, ids, fps, device_str="cpu", batch_size=2)

        assert isinstance(result, dict)
        for iid in ids:
            assert iid in result
            label, conf = result[iid]
            assert label in (0, 1)
            assert 0.0 <= conf <= 1.0

    def test_add_blur_scores_columns(self, sample_image_dir):
        """add_blur_scores should add Laplacian_Variance and Blur_Quality_Label."""
        paths = list(sample_image_dir.glob("*.jpg"))[:4]
        df = pd.DataFrame(
            {
                "Image_ID": [p.stem for p in paths],
                "File_Path": [str(p) for p in paths],
            }
        )
        result = add_blur_scores(df, model=None, laplacian_threshold=100.0)

        assert "Laplacian_Variance" in result.columns
        assert "Blur_Quality_Label" in result.columns
        assert set(result["Blur_Quality_Label"].unique()).issubset({"sharp", "blurry"})
        assert (result["Laplacian_Variance"] >= 0).all()
