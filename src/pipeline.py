"""
pipeline.py
───────────
End-to-end orchestration: ingest → preprocess → blur score → embeddings
    → dedup clustering → content clustering → recommend → save parquet.

CLI usage
---------
    python -m src.pipeline --input data/raw --output data/processed/results.parquet

All tuneable parameters are read from ``config.yaml`` and can be overridden
with explicit CLI flags.

Resume / caching
----------------
- Embedding cache:    data/processed/embeddings.npy + embedding_ids.json
- Tensor cache:       data/processed/tensors/<image_id>.pt
- Metadata parquet:   data/processed/metadata.parquet
If these files exist the pipeline skips those stages automatically.
Pass ``--force-recompute`` to ignore all caches.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

# ── ensure src/ is importable when running as __main__ ────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ingest import ingest
from src.blur_quality import add_blur_scores, build_resnet50
from src.embeddings import (
    load_clip_model,
    extract_embeddings,
    add_embeddings_to_df,
)
from src.dedup_cluster import run_dedup_clustering
from src.content_cluster import run_content_clustering, visualise_clusters
from src.recommend import recommend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# ── Required output schema columns ────────────────────────────────────────────
SCHEMA_COLUMNS = [
    "Image_ID",
    "File_Path",
    "Capture_Timestamp",
    "Laplacian_Variance",
    "Blur_Quality_Label",
    "CNN_Embedding_Vector",
    "Duplicate_Cluster_ID",
    "Similarity_Score",
    "Content_Category",
    "Keep_Recommendation",
]


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load YAML config; return empty dict if file not found."""
    p = Path(config_path)
    if not p.exists():
        logger.warning("Config file %s not found; using defaults.", p)
        return {}
    with open(p) as fh:
        return yaml.safe_load(fh) or {}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    input_dir: str | Path,
    output_path: str | Path,
    config: Dict[str, Any],
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Execute the full photo de-cluttering pipeline.

    Parameters
    ----------
    input_dir:
        Folder containing user photos.
    output_path:
        Destination parquet file for the final results.
    config:
        Parsed config.yaml dictionary.
    force_recompute:
        Ignore all caches and recompute from scratch.

    Returns
    -------
    Final results DataFrame conforming to SCHEMA_COLUMNS.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    processed_dir = output_path.parent
    processed_dir.mkdir(parents=True, exist_ok=True)

    paths_cfg = config.get("paths", {})
    emb_cfg = config.get("embeddings", {})
    blur_cfg = config.get("blur_quality", {})
    dedup_cfg = config.get("dedup_cluster", {})
    content_cfg = config.get("content_cluster", {})
    rec_cfg = config.get("recommend", {})

    emb_cache = processed_dir / Path(paths_cfg.get("embeddings_cache", "embeddings.npy")).name
    ids_cache = processed_dir / Path(paths_cfg.get("ids_cache", "embedding_ids.json")).name
    tensor_cache = processed_dir / "tensors"
    meta_parquet = processed_dir / "metadata.parquet"
    reports_dir = Path(paths_cfg.get("reports_dir", "reports"))

    # ── Step 1: Ingest ────────────────────────────────────────────────────
    if not force_recompute and meta_parquet.exists():
        logger.info("Loading cached metadata from %s …", meta_parquet)
        df = pd.read_parquet(meta_parquet)
    else:
        logger.info("=== Step 1: Ingest ===")
        df = ingest(input_dir, output_path=meta_parquet)

    if df.empty:
        logger.error("No images found in %s. Exiting.", input_dir)
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    # ── Step 2: Blur / quality scoring ───────────────────────────────────
    if "Laplacian_Variance" not in df.columns or force_recompute:
        logger.info("=== Step 2: Blur / quality scoring ===")
        blur_model = None
        checkpoint = blur_cfg.get("checkpoint")
        model_type = blur_cfg.get("model", "laplacian_only")
        if checkpoint and model_type == "resnet50":
            checkpoint_path = Path(checkpoint)
            if checkpoint_path.exists():
                logger.info("Loading ResNet50 blur model from %s …", checkpoint)
                blur_model = build_resnet50(checkpoint=checkpoint)
            else:
                logger.warning(
                    "ResNet50 checkpoint %s was not found; falling back to "
                    "Laplacian-only blur scoring.",
                    checkpoint,
                )
        df = add_blur_scores(
            df,
            model=blur_model,
            laplacian_threshold=float(blur_cfg.get("laplacian_threshold", 100.0)),
            device_str=emb_cfg.get("device", "auto"),
        )
        df.to_parquet(meta_parquet, index=False)

    # ── Step 3: CLIP embeddings ──────────────────────────────────────────
    logger.info("=== Step 3: CLIP embeddings ===")
    clip_model, clip_preprocess, clip_device = load_clip_model(
        model_name=emb_cfg.get("clip_model", "ViT-B-32"),
        pretrained=emb_cfg.get("clip_pretrained", "openai"),
        device_str=emb_cfg.get("device", "auto"),
    )
    image_ids = df["Image_ID"].tolist()
    file_paths = df["File_Path"].tolist()

    embeddings, embedding_ids = extract_embeddings(
        image_ids=image_ids,
        file_paths=file_paths,
        model=clip_model,
        preprocess=clip_preprocess,
        device=clip_device,
        batch_size=int(emb_cfg.get("batch_size", 64)),
        emb_cache_path=emb_cache,
        ids_cache_path=ids_cache,
        force_recompute=force_recompute,
    )

    df = add_embeddings_to_df(df, embeddings, embedding_ids)

    # Filter to images that have embeddings
    df_valid = df[df["CNN_Embedding_Vector"].notna()].copy()
    valid_ids = df_valid["Image_ID"].tolist()
    # Maintain consistent embedding order
    id_to_row = {iid: i for i, iid in enumerate(embedding_ids)}
    valid_embs = np.array([embeddings[id_to_row[iid]] for iid in valid_ids], dtype=np.float32)

    # ── Step 4: Near-duplicate clustering ────────────────────────────────
    logger.info("=== Step 4: Near-duplicate clustering ===")
    dedup_df = run_dedup_clustering(
        embeddings=valid_embs,
        image_ids=valid_ids,
        eps=float(dedup_cfg.get("eps", 0.15)),
        min_samples=int(dedup_cfg.get("min_samples", 2)),
        ann_neighbors=int(dedup_cfg.get("ann_neighbors", 50)),
    )
    df_valid = df_valid.merge(dedup_df, on="Image_ID", how="left")

    # ── Step 5: Content clustering ────────────────────────────────────────
    logger.info("=== Step 5: Content clustering ===")
    n_clusters: Optional[int] = content_cfg.get("n_clusters", 10)
    candidate_labels = content_cfg.get("zero_shot_labels")

    content_df = run_content_clustering(
        embeddings=valid_embs,
        image_ids=valid_ids,
        n_clusters=n_clusters,
        max_k=int(content_cfg.get("max_k", 20)),
        random_state=int(content_cfg.get("random_state", 42)),
        candidate_labels=candidate_labels,
        clip_model=clip_model,
        clip_device=clip_device,
    )
    df_valid = df_valid.merge(content_df, on="Image_ID", how="left")

    # Save cluster visualisation
    try:
        visualise_clusters(
            embeddings=valid_embs,
            labels=df_valid["Content_Category"].fillna("unknown").tolist(),
            output_path=reports_dir / "content_clusters.png",
            method="umap",
        )
    except Exception as exc:
        logger.warning("Cluster visualisation failed (non-fatal): %s", exc)

    # ── Step 6: Recommendations ───────────────────────────────────────────
    logger.info("=== Step 6: Recommendations ===")
    df_valid = recommend(
        df_valid,
        quality_delete_threshold=float(rec_cfg.get("quality_delete_threshold", 50.0)),
    )

    # ── Step 7: Enforce schema & save ────────────────────────────────────
    logger.info("=== Step 7: Saving results ===")
    for col in SCHEMA_COLUMNS:
        if col not in df_valid.columns:
            df_valid[col] = None

    result = df_valid[SCHEMA_COLUMNS].reset_index(drop=True)
    result.to_parquet(output_path, index=False)
    logger.info("Results saved to %s  (%d rows).", output_path, len(result))

    # Summary
    n_keep = (result["Keep_Recommendation"] == "Keep").sum()
    n_delete = (result["Keep_Recommendation"] == "Delete").sum()
    n_clusters_found = result["Duplicate_Cluster_ID"].nunique()
    logger.info(
        "\n══════════════════════════════════════════\n"
        "  Total images processed : %d\n"
        "  Recommended Keep       : %d\n"
        "  Recommended Delete     : %d  (%.1f%% reduction)\n"
        "  Duplicate clusters     : %d\n"
        "  Content categories     : %d\n"
        "══════════════════════════════════════════",
        len(result),
        n_keep,
        n_delete,
        100.0 * n_delete / max(1, len(result)),
        n_clusters_found,
        result["Content_Category"].nunique(),
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Automated Photo De-Clutterer — end-to-end pipeline",
    )
    p.add_argument(
        "--input", "-i",
        default=None,
        help="Input folder containing photos (default: config.yaml paths.raw_input)",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Output parquet path (default: config.yaml paths.results_parquet)",
    )
    p.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    p.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore all caches and recompute from scratch",
    )
    return p


def main(argv: Optional[list] = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)

    input_dir = args.input or config.get("paths", {}).get("raw_input", "data/raw")
    output_path = args.output or config.get("paths", {}).get(
        "results_parquet", "data/processed/results.parquet"
    )

    run_pipeline(
        input_dir=input_dir,
        output_path=output_path,
        config=config,
        force_recompute=args.force_recompute,
    )


if __name__ == "__main__":
    main()
