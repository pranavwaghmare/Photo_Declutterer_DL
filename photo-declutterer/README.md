# 📷 Automated Photo De-Clutterer

A deep-learning pipeline that ingests a folder of photos and produces
**keep / delete recommendations** based on:

1. **Blur / quality** — Laplacian variance heuristic + optional CNN / ResNet50 classifier
2. **Near-duplicate redundancy** — CLIP embeddings + DBSCAN clustering
3. **Semantic content category** — K-Means on CLIP embeddings with optional zero-shot labelling

A Streamlit UI lets you review and override recommendations; files are **never permanently deleted** — only moved to a `data/trash/` folder after explicit confirmation.

---

## Repository layout

```
photo-declutterer/
├── config.yaml               ← all tuneable parameters
├── requirements.txt
├── data/
│   ├── raw/                  ← put your photos here
│   ├── processed/            ← pipeline outputs (parquet, embeddings cache)
│   └── trash/                ← soft-deleted files land here
├── src/
│   ├── ingest.py             ← folder scan + EXIF extraction
│   ├── preprocess.py         ← resize / normalise / cache tensors
│   ├── blur_quality.py       ← Laplacian + CNN/ResNet50 blur scoring
│   ├── embeddings.py         ← CLIP embedding extraction + cache
│   ├── dedup_cluster.py      ← DBSCAN near-duplicate clustering
│   ├── content_cluster.py    ← K-Means content grouping + visualisation
│   ├── pipeline.py           ← CLI orchestrator
│   ├── recommend.py          ← composite ranking → Keep/Delete
│   └── models/
│       ├── cnn_blur.py       ← lightweight custom CNN
│       └── mobilenet_junk.py ← optional MobileNetV3 junk classifier
├── app/
│   └── review_ui.py          ← Streamlit review app
├── notebooks/
│   ├── 01_blur_quality_training.ipynb
│   ├── 02_dedup_clustering_eval.ipynb
│   └── 03_content_clustering_eval.ipynb
├── tests/
│   ├── conftest.py
│   ├── test_blur_quality.py
│   ├── test_dedup_cluster.py
│   └── test_pipeline.py
└── reports/
    └── metrics.md
```

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. Install dependencies

#### CPU-only (default)

```bash
pip install -r requirements.txt
```

#### GPU (CUDA 12.x)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> **Note**: The first pipeline run downloads ~350 MB of CLIP weights from
> OpenAI's servers.  Subsequent runs use the local cache.

---

## Running the pipeline

### On a sample folder

```bash
python -m src.pipeline --input data/raw --output data/processed/results.parquet
```

Additional options:

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `config.yaml` | Path to config file |
| `--force-recompute` | off | Ignore embedding / metadata caches |
| `--input` | `config.yaml → paths.raw_input` | Input photo directory |
| `--output` | `config.yaml → paths.results_parquet` | Output parquet path |

The pipeline is **resumable**: if embeddings are already cached in
`data/processed/embeddings.npy`, they are reused automatically.

### Expected output schema

The output parquet contains exactly these columns:

| Column | Type | Description |
|--------|------|-------------|
| `Image_ID` | str | SHA-256-based content hash |
| `File_Path` | str | Absolute path |
| `Capture_Timestamp` | datetime | EXIF timestamp or mtime |
| `Laplacian_Variance` | float | Sharpness heuristic score |
| `Blur_Quality_Label` | str | `"sharp"` or `"blurry"` |
| `CNN_Embedding_Vector` | list[float] | 512-D CLIP embedding |
| `Duplicate_Cluster_ID` | int | DBSCAN label (−1 = singleton) |
| `Similarity_Score` | float | Cosine similarity to cluster centroid |
| `Content_Category` | str | K-Means / zero-shot label |
| `Keep_Recommendation` | str | `"Keep"` or `"Delete"` |

---

## Launching the review UI

```bash
streamlit run app/review_ui.py
```

The sidebar lets you point at any `results.parquet` and any trash directory.
Clicking **Confirm and move to trash** soft-deletes only the files you have
checked — no `os.remove` is ever called.

---

## Running the tests

```bash
pytest tests/ -v
```

Tests use synthetic in-memory images; no external dataset download is required.

To include coverage:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Reproducing evaluation numbers

### CERTH blur classifier

1. Download the dataset:
   ```
   http://mklab.iti.gr/files/imageblur/CERTH_ImageBlurDataset.zip
   ```
2. Extract and organise into:
   ```
   data/certh/train/sharp/
   data/certh/train/blurry/
   data/certh/val/sharp/
   data/certh/val/blurry/
   ```
3. Open `notebooks/01_blur_quality_training.ipynb` and run all cells.

### INRIA Holidays near-duplicate evaluation

1. Download the dataset from HuggingFace:
   ```
   https://huggingface.co/datasets/randall-lab/INRIA-holidays
   ```
2. Open `notebooks/02_dedup_clustering_eval.ipynb` and follow the setup cell.

### Content clustering silhouette scores

Open `notebooks/03_content_clustering_eval.ipynb` — it works on any folder
of images; point it at your library or the CERTH / INRIA datasets.

---

## Configuration

All parameters are in `config.yaml`.  CLI flags take precedence over the file.

Key settings:

```yaml
blur_quality:
  laplacian_threshold: 100   # variance below this → blurry heuristic
  model: laplacian_only      # "laplacian_only" | "cnn" | "resnet50"

dedup_cluster:
  eps: 0.15                  # DBSCAN cosine-distance epsilon (tune this!)
  min_samples: 2             # minimum cluster size

content_cluster:
  n_clusters: 10             # null = auto-select via silhouette analysis
```

---

## Literature

1. **Pertuz et al. (2013)** — "Analysis of focus measure operators for
   shape-from-focus." *Pattern Recognition* 46(5):1415–1432.
   → Laplacian variance blur heuristic used in `blur_quality.py`.

2. **Ester et al. (1996)** — "A density-based algorithm for discovering
   clusters in large spatial databases with noise." *KDD*.
   → DBSCAN near-duplicate clustering in `dedup_cluster.py`.

3. **Radford et al. (2021)** — "Learning Transferable Visual Models From
   Natural Language Supervision." *ICML*.
   → CLIP embeddings for both dedup and content clustering
   (`embeddings.py`, `content_cluster.py`).
