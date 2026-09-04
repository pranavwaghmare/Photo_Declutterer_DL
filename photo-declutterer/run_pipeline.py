"""
Run the full pipeline on data/raw/ with settings tuned for real photos.
"""
from pathlib import Path
import sys, logging
sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

from src.pipeline import run_pipeline, load_config

config = load_config("config.yaml")

# Tuning for real photos
config["blur_quality"]["model"]      = "laplacian_only"   # fast; swap to "resnet50" for full scoring
config["blur_quality"]["checkpoint"] = None
config["blur_quality"]["laplacian_threshold"] = 80.0      # real photos: slightly lower threshold

config["dedup_cluster"]["eps"]         = 0.15             # standard cosine-distance epsilon
config["dedup_cluster"]["min_samples"] = 2

config["content_cluster"]["n_clusters"] = None            # auto-select K via silhouette
config["content_cluster"]["max_k"]      = 8               # upper bound (only 9 photos)

config["recommend"]["quality_delete_threshold"] = 80.0

config["paths"]["reports_dir"] = "reports"

result = run_pipeline(
    input_dir   = "data/raw",
    output_path = "data/processed/results.parquet",
    config      = config,
    force_recompute = False,  # reuse cached CLIP embeddings; redo clustering/recommend
)

if result.empty:
    print("No results — check that data/raw/ contains images.")
else:
    print("\n=== RESULTS ===")
    cols = ["File_Path","Laplacian_Variance","Blur_Quality_Label",
            "Duplicate_Cluster_ID","Similarity_Score","Content_Category","Keep_Recommendation"]
    display = result[cols].copy()
    display["File_Path"] = display["File_Path"].apply(lambda p: Path(p).name)
    display["Laplacian_Variance"] = display["Laplacian_Variance"].round(1)
    display["Similarity_Score"]   = display["Similarity_Score"].round(3)
    print(display.to_string(index=False))
