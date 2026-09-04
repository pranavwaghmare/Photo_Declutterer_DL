"""
preprocess.py
─────────────
Resize and normalise images for model consumption.  Results are cached as
float32 tensors (*.pt) so downstream modules can skip repeated I/O.

Public API
----------
preprocess_image(path, target_size, mean, std) → torch.Tensor  [C, H, W]
preprocess_batch(df, cache_dir, ...)           → dict[image_id, Tensor]
load_cached_tensors(cache_dir)                 → dict[image_id, Tensor]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# ImageNet statistics used by both ResNet50 and CLIP's ViT-B/32 preprocessor
_DEFAULT_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
_DEFAULT_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
_DEFAULT_SIZE: Tuple[int, int] = (224, 224)


# ─────────────────────────────────────────────────────────────────────────────
# Transform factory
# ─────────────────────────────────────────────────────────────────────────────

def build_transform(
    target_size: Tuple[int, int] = _DEFAULT_SIZE,
    mean: Tuple[float, float, float] = _DEFAULT_MEAN,
    std: Tuple[float, float, float] = _DEFAULT_STD,
) -> T.Compose:
    """
    Build a deterministic image transform pipeline.

    Parameters
    ----------
    target_size:
        (width, height) in pixels.
    mean / std:
        Per-channel normalisation statistics.

    Returns
    -------
    A ``torchvision.transforms.Compose`` object.
    """
    return T.Compose(
        [
            T.Resize(target_size),
            T.CenterCrop(target_size),
            T.ToTensor(),              # [0, 1] float, shape [C, H, W]
            T.Normalize(mean=mean, std=std),
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single-image preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(
    path: str | Path,
    target_size: Tuple[int, int] = _DEFAULT_SIZE,
    mean: Tuple[float, float, float] = _DEFAULT_MEAN,
    std: Tuple[float, float, float] = _DEFAULT_STD,
) -> Optional[torch.Tensor]:
    """
    Load an image from *path* and return a normalised [C, H, W] float32 tensor.

    Returns ``None`` if the file cannot be opened (logged as a warning).
    """
    transform = build_transform(target_size, mean, std)
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return transform(img)
    except (UnidentifiedImageError, OSError) as exc:
        logger.warning("Cannot preprocess %s: %s", path, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Batch preprocessing with disk cache
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(cache_dir: Path, image_id: str) -> Path:
    return cache_dir / f"{image_id}.pt"


def preprocess_batch(
    image_ids: List[str],
    file_paths: List[str],
    cache_dir: str | Path,
    target_size: Tuple[int, int] = _DEFAULT_SIZE,
    mean: Tuple[float, float, float] = _DEFAULT_MEAN,
    std: Tuple[float, float, float] = _DEFAULT_STD,
    force_recompute: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Preprocess a batch of images, caching each tensor individually.

    Parameters
    ----------
    image_ids:
        List of Image_ID strings (used as cache keys).
    file_paths:
        Matching list of file path strings.
    cache_dir:
        Directory where ``<image_id>.pt`` files are stored.
    force_recompute:
        If True, ignore cached files and recompute everything.

    Returns
    -------
    Dictionary mapping ``image_id → preprocessed tensor``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, torch.Tensor] = {}
    miss: List[Tuple[str, str]] = []

    for img_id, fp in zip(image_ids, file_paths):
        cp = _cache_path(cache_dir, img_id)
        if not force_recompute and cp.exists():
            result[img_id] = torch.load(cp, weights_only=True)
        else:
            miss.append((img_id, fp))

    if miss:
        logger.info("Preprocessing %d image(s) (cache misses) …", len(miss))
        transform = build_transform(target_size, mean, std)
        for img_id, fp in miss:
            try:
                with Image.open(fp) as img:
                    tensor = transform(img.convert("RGB"))
                torch.save(tensor, _cache_path(cache_dir, img_id))
                result[img_id] = tensor
            except (UnidentifiedImageError, OSError) as exc:
                logger.warning("Skipping %s during preprocessing: %s", fp, exc)

    logger.info("Preprocessed tensors available for %d image(s).", len(result))
    return result


def load_cached_tensors(cache_dir: str | Path) -> Dict[str, torch.Tensor]:
    """
    Load all previously cached tensors from *cache_dir*.

    Returns
    -------
    Dictionary mapping ``image_id → tensor``.
    """
    cache_dir = Path(cache_dir)
    tensors: Dict[str, torch.Tensor] = {}
    if not cache_dir.exists():
        return tensors
    for pt_file in sorted(cache_dir.glob("*.pt")):
        img_id = pt_file.stem
        tensors[img_id] = torch.load(pt_file, weights_only=True)
    logger.info("Loaded %d cached tensor(s) from %s.", len(tensors), cache_dir)
    return tensors
