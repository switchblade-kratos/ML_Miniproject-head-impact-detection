"""Compare HIKNet and the SVM baseline on the same held-out test samples.

Reads the result files written by train_cnn.py and svm_baseline.py (it does not
need TensorFlow) and writes results/comparison.md and figures/roc_pr.png.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.metrics import precision_recall_curve, roc_curve

from src.splits import load_split

KEYS = [("accuracy", "Accuracy"), ("precision", "Precision"), ("specificity", "Specificity"),
        ("sensitivity", "Sensitivity"), ("auc_roc", "AUC ROC"), ("auc_pr", "AUC PR")]


def pct(k, v):
    return f"{v:.3f}" if k.startswith("auc") else f"{100 * v:.1f}%"


def main():
    y = sio.loadmat("data/labels.mat")["label_impact_noimpact"].ravel().astype(int)
    split = load_split(y, 42, "results/split.json")
    y_test = y[np.array(split["test"])]

    with open("results/svm_metrics.json") as f:
        svm = json.load(f)
    cnn = None
    if os.path.exists("results/cnn_metrics.json"):
        with open("results/cnn_metrics.json") as f:
            cnn = json.load(f)

    lines = [f"Test split: n = {len(y_test)} ({int(y_test.sum())} impacts, {int((1 - y_test).sum())} false positives)", "",
             "| Model | " + " | ".join(n for _, n in KEYS) + " |",
             "|---|" + "---|" * len(KEYS)]
    if cnn:
        cells = [f"{pct(k, cnn['summary'][k]['mean'])} ± {pct(k, cnn['summary'][k]['std'])}" for k, _ in KEYS]
        lines.append(f"| HIKNet (mean ± std, {cnn['n_runs']} run(s)) | " + " | ".join(cells) + " |")
    cells = [pct(k, svm["test_metrics"][k]) for k, _ in KEYS]
    lines.append("| SVM, handcrafted features | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    with open("results/comparison.md", "w") as f:
        f.write(table + "\n")
    print(table)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    curves = [("SVM", np.load("results/svm_test_scores.npy"), "tab:red")]
    if cnn and os.path.exists("results/cnn_test_scores.npy"):
        curves.insert(0, ("HIKNet (selected run)", np.load("results/cnn_test_scores.npy"), "tab:blue"))
    for name, s, col in curves:
        fpr, tpr, _ = roc_curve(y_test, s)
        prec, rec, _ = precision_recall_curve(y_test, s)
        axes[0].plot(fpr, tpr, color=col, label=name)
        axes[1].plot(rec, prec, color=col, label=name)
    axes[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Precision-recall")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/roc_pr.png", dpi=150)


if __name__ == "__main__":
    main()
