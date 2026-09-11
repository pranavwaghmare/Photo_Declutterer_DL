"""
embeddings.py
─────────────
CLIP (ViT-B/32) embedding extraction with disk caching.

Public API
----------
load_clip_model(model_name, pretrained, device) → (model, preprocess, device)
extract_embeddings(df, model, preprocess, device, batch_size, cache_path)
    → np.ndarray  [N, D]  + updates df["CNN_Embedding_Vector"]
load_cached_embeddings(cache_path, ids_path) → (np.ndarray, list[str])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Device helper
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_clip_model(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device_str: str = "auto",
) -> Tuple[torch.nn.Module, object, torch.device]:
    """
    Load a CLIP model via ``open_clip``.

    Parameters
    ----------
    model_name:
        Architecture identifier, e.g. ``"ViT-B-32"``.
    pretrained:
        Pre-trained weights tag, e.g. ``"openai"`` or ``"laion2b_s34b_b79k"``.
    device_str:
        ``"auto"`` selects CUDA when available, otherwise CPU.

    Returns
    -------
    (model, preprocess_fn, device)
        *preprocess_fn* is the ``open_clip`` image pre-processing callable.
    """
    device = _resolve_device(device_str)
    logger.info("Loading CLIP model %s (%s) on %s …", model_name, pretrained, device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    return model, preprocess, device


# ─────────────────────────────────────────────────────────────────────────────
# Cache I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_embeddings(
    embeddings: np.ndarray,
    image_ids: List[str],
    emb_path: str | Path,
    ids_path: str | Path,
) -> None:
    """Persist embeddings array and matching ID list to disk."""
    emb_path = Path(emb_path)
    ids_path = Path(ids_path)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(emb_path), embeddings)
    with open(ids_path, "w") as fh:
        json.dump(image_ids, fh)
    logger.info("Saved embeddings %s and IDs %s.", emb_path, ids_path)


def load_cached_embeddings(
    emb_path: str | Path,
    ids_path: str | Path,
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """
    Load cached embeddings from disk.

    Returns
    -------
    (embeddings_array, image_ids_list) or (None, None) if cache is missing.
    """
    emb_path = Path(emb_path)
    ids_path = Path(ids_path)
    if not emb_path.exists() or not ids_path.exists():
        return None, None
    embeddings = np.load(str(emb_path))
    with open(ids_path) as fh:
        image_ids = json.load(fh)
    logger.info("Loaded cached embeddings: shape=%s, n_ids=%d", embeddings.shape, len(image_ids))
    return embeddings, image_ids


# ─────────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    image_ids: List[str],
    file_paths: List[str],
    model: torch.nn.Module,
    preprocess,
    device: torch.device,
    batch_size: int = 64,
    emb_cache_path: Optional[str | Path] = None,
    ids_cache_path: Optional[str | Path] = None,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract CLIP embeddings for a list of images.

    The function first checks the disk cache; images already embedded are
    skipped.  Results are persisted after extraction.

    Parameters
    ----------
    image_ids:
        List of Image_ID strings.
    file_paths:
        Matching list of file path strings.
    model:
        CLIP model (from :func:`load_clip_model`).
    preprocess:
        CLIP pre-processing transform.
    device:
        Target device.
    batch_size:
        Images per forward pass.
    emb_cache_path / ids_cache_path:
        Where to cache/load the resulting numpy array and IDs JSON.
    force_recompute:
        Ignore existing cache and recompute everything.

    Returns
    -------
    (embeddings, ordered_ids)
        *embeddings*: float32 array of shape [N, D].
        *ordered_ids*: list of Image_IDs in the same row order.
    """
    # ── Try cache ──────────────────────────────────────────────────────────
    if not force_recompute and emb_cache_path and ids_cache_path:
        cached_embs, cached_ids = load_cached_embeddings(emb_cache_path, ids_cache_path)
        if cached_embs is not None and cached_ids is not None:
            cached_id_set = set(cached_ids)
            new_ids = [i for i in image_ids if i not in cached_id_set]
            new_fps = [fp for i, fp in zip(image_ids, file_paths) if i not in cached_id_set]

            if not new_ids:
                logger.info("All %d embeddings served from cache.", len(cached_ids))
                # Return only rows requested by caller (preserving order)
                id_to_row = {iid: idx for idx, iid in enumerate(cached_ids)}
                valid = [(iid, id_to_row[iid]) for iid in image_ids if iid in id_to_row]
                out_embs = cached_embs[[r for _, r in valid]]
                out_ids = [iid for iid, _ in valid]
                return out_embs, out_ids

            # Compute only missing embeddings and merge
            new_embs, new_extracted_ids = _compute_embeddings(
                new_ids, new_fps, model, preprocess, device, batch_size
            )
            all_embs = np.concatenate([cached_embs, new_embs], axis=0)
            all_ids = cached_ids + new_extracted_ids
            if emb_cache_path and ids_cache_path:
                save_embeddings(all_embs, all_ids, emb_cache_path, ids_cache_path)

            id_to_row = {iid: idx for idx, iid in enumerate(all_ids)}
            valid = [(iid, id_to_row[iid]) for iid in image_ids if iid in id_to_row]
            return all_embs[[r for _, r in valid]], [iid for iid, _ in valid]

    # ── Full computation ────────────────────────────────────────────────────
    embeddings, extracted_ids = _compute_embeddings(
        image_ids, file_paths, model, preprocess, device, batch_size
    )
    if emb_cache_path and ids_cache_path:
        save_embeddings(embeddings, extracted_ids, emb_cache_path, ids_cache_path)
    return embeddings, extracted_ids


def _compute_embeddings(
    image_ids: List[str],
    file_paths: List[str],
    model: torch.nn.Module,
    preprocess,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, List[str]]:
    """Inner loop: compute embeddings without cache logic."""
    all_embs: List[np.ndarray] = []
    all_ids: List[str] = []

    for start in range(0, len(image_ids), batch_size):
        batch_ids = image_ids[start : start + batch_size]
        batch_fps = file_paths[start : start + batch_size]

        tensors: List[torch.Tensor] = []
        valid_ids: List[str] = []

        for img_id, fp in zip(batch_ids, batch_fps):
            try:
                img = Image.open(fp).convert("RGB")
                tensors.append(preprocess(img))
                valid_ids.append(img_id)
            except (UnidentifiedImageError, OSError) as exc:
                logger.warning("Skipping %s in embedding extraction: %s", fp, exc)

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2-normalise

        all_embs.append(feats.cpu().float().numpy())
        all_ids.extend(valid_ids)

        if (start // batch_size + 1) % 5 == 0 or start + batch_size >= len(image_ids):
            logger.info(
                "  Embeddings: %d / %d images processed.",
                min(start + batch_size, len(image_ids)),
                len(image_ids),
            )

    if not all_embs:
        return np.empty((0, 512), dtype=np.float32), []

    return np.concatenate(all_embs, axis=0).astype(np.float32), all_ids


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame integration
# ─────────────────────────────────────────────────────────────────────────────

def add_embeddings_to_df(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_ids: List[str],
) -> pd.DataFrame:
    """
    Store the raw embedding vector as a Python list in ``df["CNN_Embedding_Vector"]``.

    Rows whose Image_ID is not in *embedding_ids* get ``None``.
    """
    df = df.copy()
    id_to_emb = {iid: embeddings[i].tolist() for i, iid in enumerate(embedding_ids)}
    df["CNN_Embedding_Vector"] = df["Image_ID"].map(id_to_emb)
    return df
