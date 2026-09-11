"""
conftest.py
───────────
Shared pytest fixtures.  All fixtures use synthetic in-memory images so the
test suite runs without downloading any external dataset.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Image factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_sharp_image(width: int = 128, height: int = 128) -> Image.Image:
    """Return a synthetic sharp image (high-frequency grid pattern)."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    # Checkerboard — high spatial frequency → high Laplacian variance
    for y in range(height):
        for x in range(width):
            arr[y, x] = 255 if (x // 8 + y // 8) % 2 == 0 else 0
    return Image.fromarray(arr)


def _make_blurry_image(width: int = 128, height: int = 128) -> Image.Image:
    """Return a nearly uniform (low variance) synthetic blurry image."""
    arr = np.full((height, width, 3), fill_value=128, dtype=np.uint8)
    return Image.fromarray(arr)


def _make_random_image(width: int = 128, height: int = 128, seed: int = 0) -> Image.Image:
    """Return a random noise image (deterministic)."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_image_dir(tmp_path_factory):
    """
    Session-scoped temp directory with 8 synthetic images:
        sharp_0.jpg … sharp_3.jpg   — sharp checkerboard images
        blurry_0.jpg, blurry_1.jpg  — uniform (blurry) images
        dup_0.jpg, dup_1.jpg        — near-identical images (for dedup test)
    """
    d = tmp_path_factory.mktemp("images")

    for i in range(4):
        _make_sharp_image().save(d / f"sharp_{i}.jpg")

    for i in range(2):
        _make_blurry_image().save(d / f"blurry_{i}.jpg")

    # Near-duplicate pair: save the same image twice
    dup_img = _make_sharp_image(64, 64)
    dup_img.save(d / "dup_0.jpg")
    dup_img.save(d / "dup_1.jpg")

    return d


@pytest.fixture(scope="session")
def small_embeddings():
    """
    Session-scoped synthetic embeddings.

    Returns (embeddings, image_ids):
        embeddings: float32 array [10, 16]  (tiny dimension for speed)
        image_ids:  list of 10 string IDs
    """
    rng = np.random.default_rng(42)
    n, d = 10, 16
    raw = rng.random((n, d)).astype(np.float32)
    # L2-normalise
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    embs = raw / norms
    ids = [f"img_{i:03d}" for i in range(n)]
    return embs, ids


@pytest.fixture(scope="session")
def near_duplicate_embeddings():
    """
    Session-scoped embeddings with two obvious near-duplicate pairs.

    Returns (embeddings, image_ids, expected_clusters):
        embeddings: float32 [6, 16]
        image_ids:  6 IDs
        expected_clusters: dict mapping image_id → true group ID
    """
    rng = np.random.default_rng(0)
    d = 16

    # Two near-duplicate pairs, two distinct singletons
    base_a = rng.random(d).astype(np.float32)
    base_b = rng.random(d).astype(np.float32)
    noise = lambda: rng.random(d).astype(np.float32) * 0.01  # noqa: E731

    raw = np.array(
        [
            base_a,                      # group A
            base_a + noise(),            # group A (near-dup)
            base_b,                      # group B
            base_b + noise(),            # group B (near-dup)
            rng.random(d).astype(np.float32),  # singleton
            rng.random(d).astype(np.float32),  # singleton
        ]
    )
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    embs = raw / norms

    ids = [f"nd_{i}" for i in range(6)]
    expected = {
        "nd_0": 0, "nd_1": 0,
        "nd_2": 1, "nd_3": 1,
        "nd_4": -1, "nd_5": -1,
    }
    return embs, ids, expected
