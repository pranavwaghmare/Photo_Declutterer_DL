"""
Generate a richer synthetic photo library in data/raw/ and run the full
pipeline on it so results.parquet is ready for the Streamlit UI.

Images created (32 total):
  - 4 groups of 3 near-duplicate sharp photos  (checkerboard variants)
  - 4 blurry singletons
  - 8 "nature-like" photos  (green gradient)
  - 8 "portrait-like" photos (warm skin-tone gradient)
  - 4 "document-like" photos (white with black text pattern)
"""
import sys, logging
sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

def save(arr, name):
    Image.fromarray(arr.astype(np.uint8)).save(RAW / name)

# ── 1. Near-duplicate groups (sharp checkerboard + small colour jitter) ───────
for grp in range(4):
    base = np.zeros((256, 256, 3), dtype=np.float32)
    for y in range(256):
        for x in range(256):
            base[y, x] = 255 if (x // 16 + y // 16) % 2 == 0 else 30
    for dup in range(3):
        jitter = rng.integers(-15, 15, (256, 256, 3))
        img = np.clip(base + jitter, 0, 255)
        save(img, f"dup_grp{grp}_shot{dup}.jpg")

# ── 2. Blurry singletons (nearly uniform grey) ────────────────────────────────
for i in range(4):
    val = int(rng.integers(100, 180))
    arr = np.full((256, 256, 3), val, dtype=np.uint8)
    # tiny noise so files differ
    arr += rng.integers(0, 3, (256, 256, 3), dtype=np.uint8)
    save(arr, f"blurry_{i}.jpg")

# ── 3. Nature-like (green gradient + texture) ─────────────────────────────────
for i in range(8):
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    for y in range(256):
        arr[y, :, 0] = int(20 + y * 0.3)           # R
        arr[y, :, 1] = int(80 + y * 0.6)           # G
        arr[y, :, 2] = int(10 + y * 0.1)           # B
    noise = rng.integers(0, 30, (256, 256, 3), dtype=np.uint8)
    arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
    save(arr, f"nature_{i}.jpg")

# ── 4. Portrait-like (warm skin-tone gradient) ────────────────────────────────
for i in range(8):
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    for y in range(256):
        arr[y, :, 0] = min(255, int(180 + y * 0.2))   # R high
        arr[y, :, 1] = int(120 + y * 0.15)            # G mid
        arr[y, :, 2] = int(80 + y * 0.1)              # B low
    noise = rng.integers(0, 25, (256, 256, 3), dtype=np.uint8)
    arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
    save(arr, f"portrait_{i}.jpg")

# ── 5. Document-like (white bg + black grid lines) ───────────────────────────
for i in range(4):
    arr = np.full((256, 256, 3), 240, dtype=np.uint8)
    # horizontal lines
    for row in range(20, 256, 30):
        arr[row:row+2, 20:236] = 10
    # vertical margin line
    arr[20:236, 20:22] = 10
    noise = rng.integers(0, 8, (256, 256, 3), dtype=np.uint8)
    arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
    save(arr, f"document_{i}.jpg")

imgs = list(RAW.glob("*.jpg"))
print(f"Created {len(imgs)} images in {RAW}")

# ── Run the pipeline ──────────────────────────────────────────────────────────
print("\nRunning pipeline...")
from src.pipeline import run_pipeline, load_config
config = load_config("config.yaml")
# Override blur model to laplacian_only so we don't need the ResNet checkpoint loaded
config["blur_quality"]["model"] = "laplacian_only"
config["blur_quality"]["checkpoint"] = None
config["content_cluster"]["n_clusters"] = 5
config["dedup_cluster"]["eps"] = 0.05   # tighter: only near-identical images cluster
config["paths"]["reports_dir"] = "reports"

result = run_pipeline(
    input_dir="data/raw",
    output_path="data/processed/results.parquet",
    config=config,
    force_recompute=True,
)
print(f"\nPipeline done: {len(result)} images processed")
print(result[["Image_ID","Blur_Quality_Label","Duplicate_Cluster_ID","Content_Category","Keep_Recommendation"]].to_string())
