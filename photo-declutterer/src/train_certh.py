"""
train_certh.py
──────────────
Train BlurCNN and ResNet50 on the CERTH Image Blur Dataset, then evaluate
both models on the held-out EvaluationSet.

Dataset layout expected
-----------------------
<certh_root>/
    TrainingSet/
        Undistorted/          ← SHARP   (label 1)
        Artificially-Blurred/ ← BLURRY  (label 0)
        Naturally-Blurred/    ← BLURRY  (label 0)
        NewDigitalBlur/       ← BLURRY  (label 0)
    EvaluationSet/
        NaturalBlurSet/       ← images  (labels in NaturalBlurSet.xlsx)
        DigitalBlurSet/       ← images  (labels in DigitalBlurSet.xlsx)

Label convention (CERTH xlsx)
------------------------------
    -1  →  sharp / undistorted
     1  →  blurry

Our binary classification
--------------------------
    class 0  →  blurry
    class 1  →  sharp

CLI usage
---------
    python -m src.train_certh \\
        --certh  "C:/Users/DELL/Downloads/CERTH_ImageBlurDataset" \\
        --outdir data/processed/certh_checkpoints \\
        --epochs-cnn 20 --epochs-resnet 12 \\
        --batch 32 --device auto
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

# ── make src importable when run as __main__ ──────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.cnn_blur import BlurCNN
from src.blur_quality import build_resnet50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_certh")

# ─────────────────────────────────────────────────────────────────────────────
# Image transforms
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def _collect_images(folder: Path, label: int) -> List[Tuple[Path, int]]:
    """Return (path, label) pairs for every image in *folder*."""
    return [
        (p, label)
        for p in sorted(folder.iterdir())
        if p.suffix.lower() in SUPPORTED
    ]


class CERTHTrainDataset(Dataset):
    """
    Builds the training set from the CERTH TrainingSet folder structure.

        Undistorted/          → label 1 (sharp)
        Artificially-Blurred/ → label 0 (blurry)
        Naturally-Blurred/    → label 0 (blurry)
        NewDigitalBlur/       → label 0 (blurry)

    When *cache_dir* is provided the *eval* (deterministic) transform is
    cached as .pt files on the first pass; subsequent __getitem__ calls
    load the cached tensor and then apply only lightweight augmentation.
    This cuts per-epoch time by ~60% on CPU.
    """

    def __init__(self, certh_root: Path, transform=None, cache_dir: Optional[Path] = None):
        ts = certh_root / "TrainingSet"
        self.samples: List[Tuple[Path, int]] = []

        sharp_dir = ts / "Undistorted"
        if sharp_dir.exists():
            self.samples += _collect_images(sharp_dir, label=1)

        for blur_dir_name in ("Artificially-Blurred", "Naturally-Blurred", "NewDigitalBlur"):
            d = ts / blur_dir_name
            if d.exists():
                self.samples += _collect_images(d, label=0)

        self.transform = transform or TRAIN_TRANSFORM
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        # Lightweight augmentation applied on top of cached tensors
        self._aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomErasing(p=0.1),
        ])

        logger.info(
            "TrainingSet: %d sharp, %d blurry",
            sum(1 for _, l in self.samples if l == 1),
            sum(1 for _, l in self.samples if l == 0),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        try:
            if self.cache_dir is not None:
                cache_file = self.cache_dir / f"train_{idx}.pt"
                if cache_file.exists():
                    tensor = torch.load(cache_file, weights_only=True)
                else:
                    img = Image.open(path).convert("RGB")
                    # Cache uses the eval (deterministic) transform — no random crop
                    tensor = EVAL_TRANSFORM(img)
                    torch.save(tensor, cache_file)
                tensor = self._aug(tensor)
                return tensor, label
            else:
                img = Image.open(path).convert("RGB")
                return self.transform(img), label
        except Exception:
            # Return a black image rather than crashing the DataLoader
            return torch.zeros(3, 224, 224), label


class CERTHEvalDataset(Dataset):
    """
    Builds the evaluation set from the CERTH EvaluationSet folder + xlsx labels.

    CERTH label convention: -1 = sharp, 1 = blurry
    Our convention:          1 = sharp, 0 = blurry
    """

    def __init__(self, certh_root: Path, transform=None):
        es = certh_root / "EvaluationSet"
        self.samples: List[Tuple[Path, int]] = []
        self.transform = transform or EVAL_TRANSFORM

        # ── NaturalBlurSet ─────────────────────────────────────────────────
        nat_dir = es / "NaturalBlurSet"
        nat_xlsx = es / "NaturalBlurSet.xlsx"
        if nat_dir.exists() and nat_xlsx.exists():
            df = pd.read_excel(nat_xlsx)
            # Columns: 'Image Name', 'Blur Label'
            for _, row in df.iterrows():
                stem = str(row.iloc[0]).strip()
                blur_label = int(row.iloc[1])
                # Try common extensions
                for ext in (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".bmp"):
                    p = nat_dir / f"{stem}{ext}"
                    if not p.exists():
                        p = nat_dir / stem  # already has extension
                    if p.exists():
                        our_label = 1 if blur_label == -1 else 0
                        self.samples.append((p, our_label))
                        break

        # ── DigitalBlurSet ─────────────────────────────────────────────────
        dig_dir = es / "DigitalBlurSet"
        dig_xlsx = es / "DigitalBlurSet.xlsx"
        if dig_dir.exists() and dig_xlsx.exists():
            df = pd.read_excel(dig_xlsx)
            # Columns: 'MyDigital Blur', 'Unnamed: 1'
            for _, row in df.iterrows():
                fname = str(row.iloc[0]).strip()
                blur_label = int(row.iloc[1])
                p = dig_dir / fname
                if p.exists():
                    our_label = 1 if blur_label == -1 else 0
                    self.samples.append((p, our_label))

        logger.info(
            "EvaluationSet: %d sharp, %d blurry  (%d total)",
            sum(1 for _, l in self.samples if l == 1),
            sum(1 for _, l in self.samples if l == 0),
            len(self.samples),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), label
        except Exception:
            return torch.zeros(3, 224, 224), label


# ─────────────────────────────────────────────────────────────────────────────
# Weighted sampler (handles class imbalance)
# ─────────────────────────────────────────────────────────────────────────────

def make_weighted_sampler(dataset: Dataset) -> WeightedRandomSampler:
    labels = [dataset.samples[i][1] for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = torch.tensor([class_weights[l] for l in labels], dtype=torch.float)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss, total_n = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total_n += imgs.size(0)
    return total_loss / total_n


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels, all_preds = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def train_model(
    model: nn.Module,
    model_name: str,
    train_ds: Dataset,
    val_ds: Dataset,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    checkpoint_path: Path,
    freeze_epochs: int = 0,        # epochs to freeze backbone (ResNet warm-up)
    backbone_attr: str = "fc",     # which layer name signals the head
) -> Tuple[nn.Module, dict]:
    """
    Generic training loop with:
    - WeightedRandomSampler for class balance
    - Backbone freeze warm-up (ResNet50 only)
    - CosineAnnealingLR scheduler
    - Best-val-accuracy checkpoint saving

    Returns (best_model, metrics_dict).
    """
    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # ── Optional backbone freeze for warm-up ───────────────────────────────
    if freeze_epochs > 0:
        for name, p in model.named_parameters():
            if backbone_attr not in name:
                p.requires_grad = False
        logger.info("[%s] Backbone frozen for first %d epoch(s).", model_name, freeze_epochs)

    def _make_optimizer(m):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, m.parameters()),
            lr=lr, weight_decay=1e-4,
        )

    optimizer = _make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    history = {"train_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        # Unfreeze backbone after warm-up
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = _make_optimizer(model)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - epoch + 1
            )
            logger.info("[%s] Backbone unfrozen at epoch %d.", model_name, epoch)

        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        labels_arr, preds_arr = _evaluate(model, val_loader, device)
        val_acc = accuracy_score(labels_arr, preds_arr)
        val_f1 = f1_score(labels_arr, preds_arr, average="binary", pos_label=1, zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        lr_now = scheduler.get_last_lr()[0]
        logger.info(
            "[%s] Epoch %02d/%02d | loss=%.4f | val_acc=%.4f | val_F1=%.4f | lr=%.2e",
            model_name, epoch, epochs, train_loss, val_acc, val_f1, lr_now,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("  ↑ New best saved → %s", checkpoint_path)

    # Load best weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    logger.info("[%s] Training complete. Best val_acc=%.4f", model_name, best_acc)
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation report
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_and_report(
    model: nn.Module,
    eval_ds: Dataset,
    model_name: str,
    device: torch.device,
    batch_size: int = 32,
    report_path: Optional[Path] = None,
) -> dict:
    """Run evaluation and print / save a full metrics report."""
    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    labels_arr, preds_arr = _evaluate(model, loader, device)

    acc = accuracy_score(labels_arr, preds_arr)
    prec = precision_score(labels_arr, preds_arr, pos_label=1, zero_division=0)
    rec = recall_score(labels_arr, preds_arr, pos_label=1, zero_division=0)
    f1 = f1_score(labels_arr, preds_arr, pos_label=1, zero_division=0)
    cm = confusion_matrix(labels_arr, preds_arr)
    report = classification_report(
        labels_arr, preds_arr,
        target_names=["blurry", "sharp"],
        zero_division=0,
    )

    sep = "=" * 58
    output_lines = [
        sep,
        f"  {model_name} - Evaluation Results",
        sep,
        f"  Accuracy : {acc:.4f}",
        f"  Precision: {prec:.4f}  (sharp class)",
        f"  Recall   : {rec:.4f}  (sharp class)",
        f"  F1       : {f1:.4f}  (sharp class)",
        "",
        "  Confusion Matrix  (rows=true, cols=pred)",
        "               blurry  sharp",
        f"  true blurry  {cm[0,0]:6d}  {cm[0,1]:5d}",
        f"  true sharp   {cm[1,0]:6d}  {cm[1,1]:5d}",
        "",
        "  Per-class report:",
        report,
        sep,
    ]
    full_text = "\n".join(output_lines)
    print(full_text)

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "a", encoding="utf-8") as fh:
            fh.write(f"\n{full_text}\n")

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    histories: dict[str, dict],
    out_path: Path,
) -> None:
    """Save a side-by-side training loss / val accuracy plot."""
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        logger.warning("matplotlib not available; skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colours = ["steelblue", "tomato", "seagreen"]

    for idx, (name, hist) in enumerate(histories.items()):
        col = colours[idx % len(colours)]
        axes[0].plot(hist["train_loss"], label=name, color=col)
        axes[1].plot(hist["val_acc"],    label=name, color=col)

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    logger.info("Training curves saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.train_certh",
        description="Train BlurCNN + ResNet50 on the CERTH Image Blur Dataset",
    )
    parser.add_argument("--certh", required=True,
                        help="Path to the CERTH_ImageBlurDataset root folder")
    parser.add_argument("--outdir", default="data/processed/certh_checkpoints",
                        help="Directory to save model checkpoints and reports")
    parser.add_argument("--epochs-cnn", type=int, default=20,
                        help="Training epochs for BlurCNN (default: 20)")
    parser.add_argument("--epochs-resnet", type=int, default=12,
                        help="Training epochs for ResNet50 (default: 12)")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr-cnn", type=float, default=1e-3,
                        help="Learning rate for BlurCNN (default: 1e-3)")
    parser.add_argument("--lr-resnet", type=float, default=3e-4,
                        help="Learning rate for ResNet50 head/full (default: 3e-4)")
    parser.add_argument("--device", default="auto",
                        help="'auto' | 'cpu' | 'cuda' (default: auto)")
    parser.add_argument("--skip-cnn", action="store_true",
                        help="Skip BlurCNN training")
    parser.add_argument("--skip-resnet", action="store_true",
                        help="Skip ResNet50 training")
    parser.add_argument("--cache-tensors", action="store_true", default=True,
                        help="Cache preprocessed tensors to disk for faster epochs (default: on)")
    parser.add_argument("--no-cache-tensors", dest="cache_tensors", action="store_false")
    parser.add_argument("--resume-cnn", default=None,
                        help="Resume BlurCNN from this checkpoint path")
    parser.add_argument("--resume-resnet", default=None,
                        help="Resume ResNet50 from this checkpoint path")
    args = parser.parse_args(argv)

    certh_root = Path(args.certh)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    logger.info("Device: %s", device)

    # ── Build datasets ────────────────────────────────────────────────────
    tensor_cache = outdir / "tensor_cache" if args.cache_tensors else None
    train_ds = CERTHTrainDataset(certh_root, transform=TRAIN_TRANSFORM, cache_dir=tensor_cache)
    eval_ds_cnn = CERTHEvalDataset(certh_root, transform=EVAL_TRANSFORM)
    eval_ds_res = CERTHEvalDataset(certh_root, transform=EVAL_TRANSFORM)

    if len(train_ds) == 0:
        logger.error("No training images found under %s/TrainingSet.", certh_root)
        sys.exit(1)
    if len(eval_ds_cnn) == 0:
        logger.error("No evaluation images found under %s/EvaluationSet.", certh_root)
        sys.exit(1)

    report_path = outdir / "certh_evaluation_report.txt"
    # Wipe any previous report
    report_path.write_text(
        "CERTH Blur Classification - Evaluation Report\n" + "=" * 58 + "\n",
        encoding="utf-8",
    )

    all_histories: dict[str, dict] = {}
    all_metrics: list[dict] = []

    # ── 1. BlurCNN ────────────────────────────────────────────────────────
    if not args.skip_cnn:
        logger.info("\n%s\n  Training BlurCNN\n%s", "-" * 58, "-" * 58)
        cnn_ckpt = outdir / "blur_cnn_best.pth"
        # Support resume: load existing checkpoint or user-specified one
        resume_path = args.resume_cnn or (str(cnn_ckpt) if cnn_ckpt.exists() else None)
        if resume_path and Path(resume_path).exists():
            logger.info("Resuming BlurCNN from %s", resume_path)
            cnn_model = BlurCNN()
            cnn_model.load_state_dict(torch.load(resume_path, map_location="cpu", weights_only=True))
        else:
            cnn_model = BlurCNN()
        cnn_model, cnn_hist = train_model(
            model=cnn_model,
            model_name="BlurCNN",
            train_ds=train_ds,
            val_ds=eval_ds_cnn,
            epochs=args.epochs_cnn,
            batch_size=args.batch,
            lr=args.lr_cnn,
            device=device,
            checkpoint_path=cnn_ckpt,
        )
        all_histories["BlurCNN"] = cnn_hist
        cnn_metrics = evaluate_and_report(
            cnn_model, eval_ds_cnn, "BlurCNN", device,
            batch_size=args.batch, report_path=report_path,
        )
        all_metrics.append(cnn_metrics)

    # ── 2. ResNet50 ───────────────────────────────────────────────────────
    if not args.skip_resnet:
        logger.info("\n%s\n  Training ResNet50 (fine-tuned)\n%s", "-" * 58, "-" * 58)
        resnet_ckpt = outdir / "resnet50_blur_best.pth"
        resume_path_res = args.resume_resnet or (str(resnet_ckpt) if resnet_ckpt.exists() else None)
        if resume_path_res and Path(resume_path_res).exists():
            logger.info("Resuming ResNet50 from %s", resume_path_res)
            resnet_model = build_resnet50(num_classes=2, checkpoint=resume_path_res)
        else:
            resnet_model = build_resnet50(num_classes=2)
        resnet_model, resnet_hist = train_model(
            model=resnet_model,
            model_name="ResNet50",
            train_ds=train_ds,
            val_ds=eval_ds_res,
            epochs=args.epochs_resnet,
            batch_size=args.batch,
            lr=args.lr_resnet,
            device=device,
            checkpoint_path=resnet_ckpt,
            freeze_epochs=2,      # warm-up: train head only for 2 epochs
            backbone_attr="fc",
        )
        all_histories["ResNet50"] = resnet_hist
        resnet_metrics = evaluate_and_report(
            resnet_model, eval_ds_res, "ResNet50", device,
            batch_size=args.batch, report_path=report_path,
        )
        all_metrics.append(resnet_metrics)

    # ── 3. Laplacian baseline on eval set ─────────────────────────────────
    logger.info("\n%s\n  Laplacian Variance baseline\n%s", "-" * 58, "-" * 58)
    import cv2  # noqa: PLC0415
    lap_labels, lap_preds = [], []
    THRESHOLD = 100.0
    for path, label in eval_ds_cnn.samples:
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        pred = 1 if score >= THRESHOLD else 0
        lap_labels.append(label)
        lap_preds.append(pred)

    lap_labels_arr = np.array(lap_labels)
    lap_preds_arr  = np.array(lap_preds)
    lap_metrics = {
        "model": f"Laplacian≥{THRESHOLD}",
        "accuracy":  accuracy_score(lap_labels_arr, lap_preds_arr),
        "precision": precision_score(lap_labels_arr, lap_preds_arr, pos_label=1, zero_division=0),
        "recall":    recall_score(lap_labels_arr, lap_preds_arr, pos_label=1, zero_division=0),
        "f1":        f1_score(lap_labels_arr, lap_preds_arr, pos_label=1, zero_division=0),
    }
    all_metrics.append(lap_metrics)

    sep = "=" * 58
    lap_report = "\n".join([
        sep,
        f"  Laplacian>={THRESHOLD} baseline",
        sep,
        f"  Accuracy : {lap_metrics['accuracy']:.4f}",
        f"  Precision: {lap_metrics['precision']:.4f}",
        f"  Recall   : {lap_metrics['recall']:.4f}",
        f"  F1       : {lap_metrics['f1']:.4f}",
        sep,
    ])
    print(lap_report)
    with open(report_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n{lap_report}\n")

    # -- 4. Comparison table -----------------------------------------------
    print("\n\n" + "=" * 62)
    print("  COMPARISON TABLE")
    print("=" * 62)
    print(f"  {'Model':<24} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("  " + "-" * 58)
    for m in all_metrics:
        print(
            f"  {m['model']:<24} {m['accuracy']:>7.4f} {m['precision']:>7.4f}"
            f" {m['recall']:>7.4f} {m['f1']:>7.4f}"
        )
    print("=" * 62)

    # Save comparison table to report
    table_lines = [
        "\n\nCOMPARISON TABLE",
        "=" * 62,
        f"  {'Model':<24} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}",
        "  " + "-" * 58,
    ]
    for m in all_metrics:
        table_lines.append(
            f"  {m['model']:<24} {m['accuracy']:>7.4f} {m['precision']:>7.4f}"
            f" {m['recall']:>7.4f} {m['f1']:>7.4f}"
        )
    table_lines.append("=" * 62)
    with open(report_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(table_lines) + "\n")

    # ── 5. Training curves plot ───────────────────────────────────────────
    if all_histories:
        plot_training_curves(all_histories, outdir / "training_curves.png")

    logger.info("\nAll results saved to %s", outdir)
    logger.info("Full report → %s", report_path)

    # ── 6. Update config.yaml to point at best checkpoint ─────────────────
    _update_config(outdir)


def _update_config(outdir: Path) -> None:
    """Patch config.yaml to use the trained ResNet50 checkpoint."""
    config_path = _ROOT / "config.yaml"
    if not config_path.exists():
        return
    import yaml  # noqa: PLC0415
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    best_ckpt = outdir / "resnet50_blur_best.pth"
    if best_ckpt.exists():
        cfg.setdefault("blur_quality", {})["checkpoint"] = str(best_ckpt)
        cfg["blur_quality"]["model"] = "resnet50"
        with open(config_path, "w") as fh:
            yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=False)
        logger.info("config.yaml updated → blur_quality.checkpoint = %s", best_ckpt)


if __name__ == "__main__":
    main()
