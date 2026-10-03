"""Evaluation metrics used for every model (same definitions as the paper)."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_metrics(y_true, y_pred, y_score=None):
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    def ratio(a, b):
        return a / b if b else float("nan")

    out = {
        "accuracy": ratio(tp + tn, tp + tn + fp + fn),
        "precision": ratio(tp, tp + fp),
        "specificity": ratio(tn, tn + fp),
        "sensitivity": ratio(tp, tp + fn),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
    if y_score is not None:
        out["auc_roc"] = float(roc_auc_score(y_true, y_score))
        out["auc_pr"] = float(average_precision_score(y_true, y_score))
    return out


def format_metrics(m):
    keys = ["accuracy", "precision", "specificity", "sensitivity", "auc_roc", "auc_pr"]
    return "  ".join(f"{k}={m[k]:.3f}" for k in keys if k in m)
