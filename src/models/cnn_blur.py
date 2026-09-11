"""
models/cnn_blur.py
──────────────────
Lightweight custom CNN for binary Sharp / Blurry classification.

Architecture (small enough to train quickly on CPU):
    Conv(3→16, 3×3) → BN → ReLU → MaxPool
    Conv(16→32, 3×3) → BN → ReLU → MaxPool
    Conv(32→64, 3×3) → BN → ReLU → AdaptiveAvgPool(4×4)
    FC(1024→128) → ReLU → Dropout
    FC(128→2)

Input: normalised RGB tensor [B, 3, 224, 224]
Output: logits [B, 2]  (class 0 = blurry, class 1 = sharp)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BlurCNN(nn.Module):
    """Lightweight binary blur classifier."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.4) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.classifier(self.features(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax class probabilities — shape [B, num_classes]."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_blur_cnn(checkpoint: str | None = None) -> BlurCNN:
    """
    Instantiate a :class:`BlurCNN`, optionally loading weights from *checkpoint*.

    Parameters
    ----------
    checkpoint:
        Path to a ``torch.save``'d state dict. Pass ``None`` for random init.

    Returns
    -------
    BlurCNN in eval mode.
    """
    model = BlurCNN()
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.eval()
    return model
