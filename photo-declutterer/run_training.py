"""
CERTH training launcher — Phase 3: finish ResNet50 + generate full report.

Resumes ResNet50 from its best checkpoint (epoch 3, val_acc=0.847)
and runs 3 more epochs, then evaluates all three models and prints
the comparison table.
"""
import sys
sys.path.insert(0, ".")
from src.train_certh import main

main([
    "--certh",          r"C:\Users\DELL\Downloads\CERTH_ImageBlurDataset",
    "--outdir",         "data/processed/certh_checkpoints",
    "--epochs-cnn",     "10",
    "--epochs-resnet",  "3",    # 3 more epochs from current best
    "--batch",          "32",
    "--device",         "auto",
    "--skip-cnn",               # CNN already done
    "--cache-tensors",
])
