"""
models/mobilenet_junk.py  (optional stretch goal)
──────────────────────────────────────────────────
MobileNetV3-Small fine-tuned for binary Junk / Real classification.

"Junk" images: screenshots, memes, low-resolution web images, scan artifacts.
"Real" images: genuine photographs worth considering for retention.

This module is *optional* and skippable — the main pipeline will import it
only when ``config.yaml`` enables it.  The model definition is included here
so the architecture is ready for training as soon as labelled data is
available.

Input:  normalised RGB tensor  [B, 3, 224, 224]
Output: logits                 [B, 2]  (class 0 = junk, class 1 = real)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models


class MobileNetJunkClassifier(nn.Module):
    """
    MobileNetV3-Small with a custom binary head.

    Parameters
    ----------
    pretrained:
        Use ImageNet-pretrained weights for the backbone.
    freeze_backbone:
        Freeze all layers except the custom classification head during
        early-stage fine-tuning.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        weights = tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.mobilenet_v3_small(weights=weights)

        # Replace the classifier head
        in_features: int = backbone.classifier[3].in_features  # type: ignore[index]
        backbone.classifier[3] = nn.Linear(in_features, num_classes)
        self.backbone = backbone

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities — shape [B, 2]."""
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=-1)

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (call after initial warm-up epochs)."""
        for param in self.parameters():
            param.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_junk_classifier(
    checkpoint: Optional[str | Path] = None,
    pretrained: bool = True,
) -> MobileNetJunkClassifier:
    """
    Build a :class:`MobileNetJunkClassifier`, optionally loading *checkpoint*.

    Parameters
    ----------
    checkpoint:
        Path to a ``torch.save``'d state dict; ``None`` for fresh weights.
    pretrained:
        Download ImageNet weights for the backbone when *checkpoint* is None.

    Returns
    -------
    Model in eval mode.
    """
    model = MobileNetJunkClassifier(pretrained=(checkpoint is None and pretrained))
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.eval()
    return model
