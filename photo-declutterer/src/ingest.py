"""
ingest.py
─────────
Recursively scans a folder for images, reads EXIF metadata, assigns a
stable UUID per file, and returns a pandas DataFrame that becomes the
backbone table every downstream module joins into.

Output columns:
    Image_ID            – UUID string (stable: derived from file content hash)
    File_Path           – absolute path string
    Capture_Timestamp   – datetime parsed from EXIF or file mtime fallback
    Width               – pixel width
    Height              – pixel height
    File_Size_Bytes     – raw file size
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from PIL import Image, ExifTags, UnidentifiedImageError

logger = logging.getLogger(__name__)

# EXIF tag name → numeric ID mapping (built at import time)
_EXIF_TAG_ID: dict[str, int] = {v: k for k, v in ExifTags.TAGS.items()}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp", ".heic"}
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _file_hash(path: Path, chunk_size: int = 65536) -> str:
    """Return a SHA-256 hex digest (first 32 chars) of the full file content."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()[:32]


def _make_image_id(path: Path) -> str:
    """
    Derive a stable Image_ID from the file's content hash so that the same
    photo always gets the same ID regardless of where it lives on disk.
    """
    return _file_hash(path)


def _read_exif_timestamp(path: Path) -> Optional[datetime]:
    """
    Try to extract DateTimeOriginal (or DateTime) from EXIF.

    Returns ``None`` if the image has no EXIF or the tag is absent/malformed.
    """
    try:
        img = Image.open(path)
        exif_data = img._getexif()  # type: ignore[attr-defined]
        if exif_data is None:
            return None
        for tag_name in ("DateTimeOriginal", "DateTime", "DateTimeDigitized"):
            tag_id = _EXIF_TAG_ID.get(tag_name)
            if tag_id and tag_id in exif_data:
                raw = exif_data[tag_id]
                try:
                    return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    pass
    except Exception:  # noqa: BLE001
        pass
    return None


def _image_dimensions(path: Path) -> tuple[int, int]:
    """Return (width, height) without fully decoding the image."""
    try:
        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception:
        return (0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def scan_folder(
    root: str | Path,
    extensions: frozenset[str] = SUPPORTED_EXTENSIONS,
) -> List[Path]:
    """
    Recursively yield all image files under *root* with a supported extension.

    Parameters
    ----------
    root:
        Top-level directory to search.
    extensions:
        Set of lowercase extensions (including the leading dot) to include.

    Returns
    -------
    Sorted list of :class:`pathlib.Path` objects.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")

    found: List[Path] = []
    for dirpath, _dirs, filenames in os.walk(root):
        for fname in filenames:
            p = Path(dirpath) / fname
            if p.suffix.lower() in extensions:
                found.append(p)

    found.sort()
    logger.info("Scanned %s → found %d image file(s).", root, len(found))
    return found


def build_metadata_df(
    paths: List[Path],
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Build the backbone metadata DataFrame from a list of image paths.

    Parameters
    ----------
    paths:
        List of image file paths (as returned by :func:`scan_folder`).
    show_progress:
        Log progress every 100 images.

    Returns
    -------
    DataFrame with columns:
        Image_ID, File_Path, Capture_Timestamp, Width, Height, File_Size_Bytes
    """
    records = []
    for i, p in enumerate(paths, 1):
        if show_progress and i % 100 == 0:
            logger.info("  Ingesting image %d / %d …", i, len(paths))

        try:
            image_id = _make_image_id(p)
            ts = _read_exif_timestamp(p)
            if ts is None:
                ts = datetime.fromtimestamp(p.stat().st_mtime)
            width, height = _image_dimensions(p)
            file_size = p.stat().st_size

            records.append(
                {
                    "Image_ID": image_id,
                    "File_Path": str(p.resolve()),
                    "Capture_Timestamp": ts,
                    "Width": width,
                    "Height": height,
                    "File_Size_Bytes": file_size,
                }
            )
        except (OSError, UnidentifiedImageError) as exc:
            logger.warning("Skipping %s: %s", p, exc)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Drop true duplicates (same hash = same file content)
    before = len(df)
    df = df.drop_duplicates(subset="Image_ID").reset_index(drop=True)
    if len(df) < before:
        logger.info("Dropped %d content-identical duplicates.", before - len(df))

    logger.info("Ingested %d unique images.", len(df))
    return df


def ingest(
    input_dir: str | Path,
    output_path: Optional[str | Path] = None,
    extensions: frozenset[str] = SUPPORTED_EXTENSIONS,
) -> pd.DataFrame:
    """
    Full ingest pipeline: scan → build DataFrame → optionally save parquet.

    Parameters
    ----------
    input_dir:
        Root directory containing user photos.
    output_path:
        If provided, save the DataFrame as a parquet file to this path.
    extensions:
        Image extensions to include.

    Returns
    -------
    Metadata DataFrame.
    """
    paths = scan_folder(input_dir, extensions=extensions)
    df = build_metadata_df(paths)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        logger.info("Saved metadata to %s", out)

    return df
