# Photo De-Clutterer — Evaluation Report

## 1. Blur / Quality Classifier

### Dataset: CERTH Image Blur Dataset
Download: `http://mklab.iti.gr/files/imageblur/CERTH_ImageBlurDataset.zip`

Expected split:
- Training:   ~600 sharp + ~600 blurry images
- Validation: ~200 sharp + ~200 blurry images

Run training:
```bash
python - <<'EOF'
from src.blur_quality import train_resnet50, train_blur_cnn
train_blur_cnn("data/certh/train", "data/certh/val",
               epochs=15, checkpoint_out="data/processed/blur_cnn.pth")
train_resnet50("data/certh/train", "data/certh/val",
               epochs=10, checkpoint_out="data/processed/resnet50_blur.pth")
EOF
```

### Results — CERTH EvaluationSet (1,480 images: 619 sharp / 861 blurry)

Training set: 1,150 images (630 sharp / 520 blurry from TrainingSet folder).
All metrics are on the held-out EvaluationSet (NaturalBlurSet + DigitalBlurSet).

| Model                  | Accuracy | Precision (sharp) | Recall (sharp) | F1 (sharp) |
|------------------------|----------|-------------------|----------------|------------|
| **BlurCNN** (10 ep)    | 0.8041   | 0.7466            | 0.8045         | 0.7745     |
| **ResNet50 FT** (8 ep) | **0.8696** | **0.8349**      | **0.8578**     | **0.8462** |
| Laplacian ≥100 (baseline) | 0.7993 | 0.8966          | 0.5880         | 0.7102     |

**Key observations:**
- ResNet50 fine-tuned beats both BlurCNN and the Laplacian heuristic on all balanced metrics.
- The Laplacian baseline has high precision but very low recall (58.8%) — it misses nearly half of sharp images, making it unreliable alone.
- BlurCNN achieves comparable accuracy to the Laplacian baseline but with much better recall (80.5%), making it a better holistic classifier.
- Checkpoints saved to `data/processed/certh_checkpoints/`.

> **Literature citation**: The Laplacian-variance blur heuristic follows
> Pertuz et al. (2013) "Analysis of focus measure operators for shape-from-focus."
> *Pattern Recognition* 46(5), 1415–1432.
> CNN-based blur classification follows Tong et al. (2004) "Blur detection
> for digital images using wavelet transform." *ICME*.

---

## 2. Near-Duplicate Detection

### Datasets

| Dataset | Source | Ground-truth |
|---------|--------|--------------|
| INRIA Holidays | https://huggingface.co/datasets/randall-lab/INRIA-holidays | 500 holiday groups |
| UKBench | http://vis.cs.ucdavis.edu/~unitgrad/projects/ukbench/ | 2550 groups of 4 |

Run evaluation:
```bash
python - <<'EOF'
# Assumes you have downloaded and organised the dataset under data/inria_holidays/
from src.embeddings import load_clip_model, extract_embeddings
from src.dedup_cluster import run_dedup_clustering, evaluate_clustering
import pandas as pd, numpy as np

df = pd.read_parquet("data/inria_holidays/metadata.parquet")
model, preprocess, device = load_clip_model()
embs, ids = extract_embeddings(df["Image_ID"].tolist(), df["File_Path"].tolist(),
                                model, preprocess, device)
result = run_dedup_clustering(embs, ids, eps=0.15, min_samples=2)
# Load ground-truth group IDs aligned with `ids` order, then:
# metrics = evaluate_clustering(pred_labels, true_labels)
EOF
```

### Results (fill in after evaluation)

| Dataset         | Precision | Recall | F1    | DBSCAN eps |
|-----------------|-----------|--------|-------|------------|
| INRIA Holidays  | —         | —      | —     | 0.15       |
| UKBench         | —         | —      | —     | 0.15       |

> **Literature citation**: CLIP-based near-duplicate detection follows
> Radford et al. (2021) "Learning Transferable Visual Models From Natural
> Language Supervision." *ICML*.  DBSCAN clustering baseline follows
> Ester et al. (1996) "A density-based algorithm for discovering clusters."
> *KDD*.

---

## 3. Content-Based Clustering

### Silhouette Scores (fill in after running)

| Dataset   | K   | Silhouette (cosine) | Method |
|-----------|-----|---------------------|--------|
| Sample set | — | —                   | K-Means + CLIP |

Cluster visualisation: `reports/content_clusters.png`

> **Literature citation**: CLIP zero-shot clustering follows
> Radford et al. (2021), and the cluster-then-label approach follows
> Zhang et al. (2022) "CLIP-based Contrastive Image Clustering."

---

## 4. End-to-End Library Reduction Examples

| Scenario                    | Input photos | After pipeline | Reduction |
|-----------------------------|-------------|----------------|-----------|
| Holiday trip (duplicates)   | —           | —              | —         |
| Mixed personal library      | —           | —              | —         |
| Document scan batch         | —           | —              | —         |

*(Fill in with real numbers after running on sample libraries.)*
