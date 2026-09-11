"""
blur_quality.py
───────────────
Blur / quality assessment pipeline.

Three tiers (all optional / composable):

1. **Laplacian variance** (fast, no model) — OpenCV-based heuristic.
2. **BlurCNN** (lightweight custom CNN) — trained on CERTH Image Blur Dataset.
3. **ResNet50** (fine-tuned ImageNet model) — stronger classifier.

Public API
----------
score_laplacian(path) → float
score_batch(df, model, device, batch_size) → pd.DataFrame  (adds columns)
train_resnet50(train_dir, val_dir, ...)    → nn.Module
train_blur_cnn(train_dir, val_dir, ...)    → BlurCNN
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tv_models
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.models.cnn_blur import BlurCNN, build_blur_cnn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Laplacian variance (heuristic, always available)
# ─────────────────────────────────────────────────────────────────────────────

def score_laplacian(path: str | Path) -> float:
    """
    Compute the variance of the Laplacian as a sharpness measure.

    Higher → sharper.  A common threshold is ~100; below that the image is
    likely blurry.

    Parameters
    ----------
    path:
        Path to an image file readable by OpenCV.

    Returns
    -------
    Non-negative float (0.0 on read failure).
    """
    img = cv2.imread(str(path))
    if img is None:
        logger.warning("cv2.imread returned None for %s", path)
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def score_laplacian_batch(
    image_ids: List[str],
    file_paths: List[str],
) -> Dict[str, float]:
    """Score all images and return a dict mapping image_id → Laplacian_Variance."""
    return {
        img_id: score_laplacian(fp)
        for img_id, fp in zip(image_ids, file_paths)
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset helper for CERTH-style directory layout
# ─────────────────────────────────────────────────────────────────────────────

class BlurImageDataset(Dataset):
    """
    Expects a directory with two sub-folders:
        <root>/sharp/   — class 1 (sharp)
        <root>/blurry/  — class 0 (blurry)
    """

    LABEL_MAP = {"blurry": 0, "sharp": 1}
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(self, root: str | Path, transform: Optional[transforms.Compose] = None):
        self.root = Path(root)
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.samples: List[Tuple[Path, int]] = []
        for cls_name, label in self.LABEL_MAP.items():
            cls_dir = self.root / cls_name
            if cls_dir.exists():
                for p in cls_dir.iterdir():
                    if p.suffix.lower() in self.EXTENSIONS:
                        self.samples.append((p, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ─────────────────────────────────────────────────────────────────────────────
# 3. ResNet50 factory
# ─────────────────────────────────────────────────────────────────────────────

def build_resnet50(num_classes: int = 2, checkpoint: Optional[str] = None) -> nn.Module:
    """
    Load ImageNet-pretrained ResNet50 and replace the final FC layer.

    Parameters
    ----------
    num_classes:
        Output classes (2 for sharp/blurry).
    checkpoint:
        Path to a fine-tuned state dict; None → pretrained ImageNet weights only.

    Returns
    -------
    ResNet50 in eval mode.
    """
    weights = tv_models.ResNet50_Weights.IMAGENET1K_V1
    model = tv_models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training routines (called from training notebooks or CLI)
# ─────────────────────────────────────────────────────────────────────────────

def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)  # type: ignore[arg-type]


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return (accuracy, all_labels, all_preds)."""
    model.eval()
    all_labels, all_preds = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)
    acc = float((labels_arr == preds_arr).mean())
    return acc, labels_arr, preds_arr


def train_resnet50(
    train_dir: str | Path,
    val_dir: str | Path,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    checkpoint_out: Optional[str | Path] = None,
    device_str: str = "auto",
) -> nn.Module:
    """
    Fine-tune ResNet50 on a CERTH-style blur dataset.

    Parameters
    ----------
    train_dir / val_dir:
        Directories with ``sharp/`` and ``blurry/`` sub-folders.
    epochs:
        Number of training epochs.
    batch_size:
        Mini-batch size.
    lr:
        Initial learning rate for Adam.
    checkpoint_out:
        Where to save the best model state dict.
    device_str:
        ``"auto"`` picks CUDA if available, else CPU.

    Returns
    -------
    Fine-tuned ResNet50 in eval mode.
    """
    device = _resolve_device(device_str)
    model = build_resnet50().to(device)

    # Freeze backbone; only train the new head first
    for name, p in model.named_parameters():
        p.requires_grad = "fc" in name

    train_ds = BlurImageDataset(train_dir)
    val_ds = BlurImageDataset(val_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        # Unfreeze all after epoch 2 for full fine-tuning
        if epoch == 3:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=lr * 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, _, _ = _evaluate(model, val_loader, device)
        scheduler.step()
        logger.info("Epoch %d/%d | train_loss=%.4f | val_acc=%.4f", epoch, epochs, train_loss, val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            if checkpoint_out:
                Path(checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_out)

    logger.info("Best validation accuracy: %.4f", best_acc)
    model.eval()
    return model


def train_blur_cnn(
    train_dir: str | Path,
    val_dir: str | Path,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    checkpoint_out: Optional[str | Path] = None,
    device_str: str = "auto",
) -> BlurCNN:
    """
    Train the lightweight :class:`~src.models.cnn_blur.BlurCNN` from scratch.

    Same parameter conventions as :func:`train_resnet50`.
    """
    device = _resolve_device(device_str)
    model = BlurCNN().to(device)

    train_ds = BlurImageDataset(train_dir)
    val_ds = BlurImageDataset(val_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, _, _ = _evaluate(model, val_loader, device)
        scheduler.step()
        logger.info("Epoch %d/%d | train_loss=%.4f | val_acc=%.4f", epoch, epochs, train_loss, val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            if checkpoint_out:
                Path(checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_out)

    logger.info("Best validation accuracy: %.4f", best_acc)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5. Inference helpers (used by pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

_INFER_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


@torch.no_grad()
def score_batch_model(
    model: nn.Module,
    image_ids: List[str],
    file_paths: List[str],
    device_str: str = "auto",
    batch_size: int = 32,
) -> Dict[str, Tuple[int, float]]:
    """
    Run *model* over all images and return per-image predictions.

    Returns
    -------
    Dict mapping image_id → (predicted_label, confidence_score)
        label: 0=blurry, 1=sharp
        confidence: softmax probability for the predicted class
    """
    device = _resolve_device(device_str)
    model = model.to(device).eval()

    results: Dict[str, Tuple[int, float]] = {}
    for start in range(0, len(image_ids), batch_size):
        batch_ids = image_ids[start : start + batch_size]
        batch_fps = file_paths[start : start + batch_size]

        tensors = []
        valid_ids = []
        for img_id, fp in zip(batch_ids, batch_fps):
            try:
                img = Image.open(fp).convert("RGB")
                tensors.append(_INFER_TRANSFORM(img))
                valid_ids.append(img_id)
            except Exception as exc:
                logger.warning("Skipping %s in model inference: %s", fp, exc)

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)
        probs = torch.softmax(model(batch_tensor), dim=1).cpu().numpy()
        for img_id, prob_row in zip(valid_ids, probs):
            pred = int(np.argmax(prob_row))
            conf = float(prob_row[pred])
            results[img_id] = (pred, conf)

    return results


def add_blur_scores(
    df: pd.DataFrame,
    model: Optional[nn.Module] = None,
    laplacian_threshold: float = 100.0,
    device_str: str = "auto",
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Augment *df* with blur/quality columns.

    Columns added / overwritten:
        Laplacian_Variance  – float
        Blur_Quality_Label  – "sharp" | "blurry"  (heuristic or model)

    Parameters
    ----------
    df:
        Must contain ``Image_ID`` and ``File_Path`` columns.
    model:
        If provided, use model predictions for ``Blur_Quality_Label``;
        otherwise fall back to the Laplacian threshold.
    laplacian_threshold:
        Heuristic cutoff: variance < threshold → "blurry".

    Returns
    -------
    Modified copy of *df*.
    """
    df = df.copy()

    image_ids = df["Image_ID"].tolist()
    file_paths = df["File_Path"].tolist()

    # Laplacian variance (always computed — cheap, and used in recommend.py)
    logger.info("Computing Laplacian variance for %d images …", len(df))
    lap_scores = score_laplacian_batch(image_ids, file_paths)
    df["Laplacian_Variance"] = df["Image_ID"].map(lap_scores).fillna(0.0)

    if model is not None:
        logger.info("Running blur model inference …")
        model_preds = score_batch_model(model, image_ids, file_paths, device_str, batch_size)
        def _resolve_label(iid: str, lap_val: float) -> str:
            if iid in model_preds:
                return "sharp" if model_preds[iid][0] == 1 else "blurry"
            return "sharp" if lap_val >= laplacian_threshold else "blurry"

        df["Blur_Quality_Label"] = [
            _resolve_label(iid, lap)
            for iid, lap in zip(df["Image_ID"], df["Laplacian_Variance"])
        ]
    else:
        df["Blur_Quality_Label"] = df["Laplacian_Variance"].map(
            lambda v: "sharp" if v >= laplacian_threshold else "blurry"
        )

    return df
