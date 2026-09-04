"""
tests/test_models.py
────────────────────
Unit tests for model architectures.
"""

from __future__ import annotations

import torch
import pytest

from src.models.cnn_blur import BlurCNN, build_blur_cnn
from src.models.mobilenet_junk import MobileNetJunkClassifier, build_junk_classifier


class TestModels:
    def test_blur_cnn_forward(self):
        model = BlurCNN()
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 2)

    def test_mobilenet_junk_forward(self):
        model = MobileNetJunkClassifier(pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 2)

    def test_mobilenet_junk_predict_proba(self):
        model = MobileNetJunkClassifier(pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        probs = model.predict_proba(x)
        assert probs.shape == (2, 2)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)

    def test_build_junk_classifier(self):
        model = build_junk_classifier(pretrained=False)
        assert not model.training
