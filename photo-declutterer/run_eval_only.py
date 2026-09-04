"""
Evaluate both saved checkpoints on the CERTH EvaluationSet and print
the full comparison table. No training — just inference.
"""
import sys
sys.path.insert(0, ".")

import logging
import numpy as np
import cv2
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from src.models.cnn_blur import BlurCNN
from src.blur_quality import build_resnet50
from src.train_certh import CERTHEvalDataset, _evaluate, EVAL_TRANSFORM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("eval")

CERTH  = Path(r"C:\Users\DELL\Downloads\CERTH_ImageBlurDataset")
OUTDIR = Path("data/processed/certh_checkpoints")
REPORT = OUTDIR / "certh_evaluation_report.txt"

device = torch.device("cpu")

eval_ds = CERTHEvalDataset(CERTH, transform=EVAL_TRANSFORM)
loader  = DataLoader(eval_ds, batch_size=32, shuffle=False, num_workers=0)
log.info("Eval set: %d samples", len(eval_ds))

all_metrics = []

def report_model(model, name):
    log.info("Evaluating %s ...", name)
    labels, preds = _evaluate(model, loader, device)
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, pos_label=1, zero_division=0)
    rec  = recall_score(labels, preds, pos_label=1, zero_division=0)
    f1   = f1_score(labels, preds, pos_label=1, zero_division=0)
    cm   = confusion_matrix(labels, preds)
    rep  = classification_report(labels, preds, target_names=["blurry","sharp"], zero_division=0)
    lines = [
        "=" * 58,
        f"  {name} - Evaluation Results",
        "=" * 58,
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
        rep,
        "=" * 58,
    ]
    txt = "\n".join(lines)
    print(txt)
    all_metrics.append({"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
    return txt

texts = []

# ── BlurCNN ──────────────────────────────────────────────────────────────────
cnn_ckpt = OUTDIR / "blur_cnn_best.pth"
cnn = BlurCNN()
cnn.load_state_dict(torch.load(cnn_ckpt, map_location="cpu", weights_only=True))
cnn.eval()
texts.append(report_model(cnn, "BlurCNN"))

# ── ResNet50 ─────────────────────────────────────────────────────────────────
res_ckpt = OUTDIR / "resnet50_blur_best.pth"
res = build_resnet50(num_classes=2, checkpoint=str(res_ckpt))
res.eval()
texts.append(report_model(res, "ResNet50 (fine-tuned)"))

# ── Laplacian baseline ───────────────────────────────────────────────────────
log.info("Running Laplacian baseline ...")
THRESHOLD = 100.0
lap_labels, lap_preds = [], []
for path, label in eval_ds.samples:
    img = cv2.imread(str(path))
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lap_labels.append(label)
    lap_preds.append(1 if score >= THRESHOLD else 0)

la, lp = np.array(lap_labels), np.array(lap_preds)
lap_m = {
    "model":     f"Laplacian>={THRESHOLD}",
    "accuracy":  accuracy_score(la, lp),
    "precision": precision_score(la, lp, pos_label=1, zero_division=0),
    "recall":    recall_score(la, lp, pos_label=1, zero_division=0),
    "f1":        f1_score(la, lp, pos_label=1, zero_division=0),
}
all_metrics.append(lap_m)
lap_txt = "\n".join([
    "=" * 58,
    f"  Laplacian>={THRESHOLD} baseline",
    "=" * 58,
    f"  Accuracy : {lap_m['accuracy']:.4f}",
    f"  Precision: {lap_m['precision']:.4f}",
    f"  Recall   : {lap_m['recall']:.4f}",
    f"  F1       : {lap_m['f1']:.4f}",
    "=" * 58,
])
print(lap_txt)
texts.append(lap_txt)

# ── Comparison table ─────────────────────────────────────────────────────────
print("\n\n" + "=" * 62)
print("  COMPARISON TABLE")
print("=" * 62)
print(f"  {'Model':<26} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print("  " + "-" * 58)
for m in all_metrics:
    print(f"  {m['model']:<26} {m['accuracy']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f}")
print("=" * 62)

# ── Save full report ─────────────────────────────────────────────────────────
OUTDIR.mkdir(parents=True, exist_ok=True)
with open(REPORT, "w", encoding="utf-8") as fh:
    fh.write("CERTH Blur Classification - Evaluation Report\n" + "=" * 58 + "\n\n")
    for t in texts:
        fh.write(t + "\n\n")
    fh.write("\nCOMPARISON TABLE\n" + "=" * 62 + "\n")
    fh.write(f"  {'Model':<26} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}\n")
    fh.write("  " + "-" * 58 + "\n")
    for m in all_metrics:
        fh.write(f"  {m['model']:<26} {m['accuracy']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f}\n")
    fh.write("=" * 62 + "\n")

print(f"\nFull report saved -> {REPORT}")
