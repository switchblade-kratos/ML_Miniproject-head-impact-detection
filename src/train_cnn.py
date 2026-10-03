"""Train HIKNet and evaluate it once on the held-out test split.

Training uses the train split; the validation split drives early stopping
(patience 5, best epoch restored). The test split is only used at the end.
With --runs N the whole procedure is repeated with different seeds (the paper
averaged 10 runs because results depend on the weight initialization).

Outputs:
    checkpoints/hiknet_run<i>.keras   best model of each run
    checkpoints/hiknet_best.keras     run with the lowest validation loss (used by the demo)
    results/cnn_metrics.json          per-run and mean/std test metrics
    results/cnn_test_scores.npy       test probabilities of the selected run
    figures/training_curves.png       loss curves of the selected run
"""

import argparse
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from tensorflow import keras

from src.hiknet import build_hiknet
from src.metrics import compute_metrics, format_metrics
from src.preprocess import standardize
from src.splits import load_split

METRIC_KEYS = ["accuracy", "precision", "specificity", "sensitivity", "auc_roc", "auc_pr"]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    x = sio.loadmat(os.path.join(a.data_dir, "data.mat"))["data"]
    y = sio.loadmat(os.path.join(a.data_dir, "labels.mat"))["label_impact_noimpact"].ravel().astype("float32")
    split = load_split(y, a.seed, "results/split.json")
    xs = standardize(x)
    tr, va, te = (np.array(split[k]) for k in ("train", "val", "test"))

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    runs, best_val, best_run = [], np.inf, None
    for r in range(a.runs):
        keras.utils.set_random_seed(a.seed + r)
        model = build_hiknet()
        if r == 0:
            model.summary()
        ckpt = f"checkpoints/hiknet_run{r}.keras"
        hist = model.fit(
            xs[tr], y[tr], validation_data=(xs[va], y[va]),
            epochs=a.max_epochs, batch_size=a.batch_size, verbose=2,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
            ])
        val_loss = float(min(hist.history["val_loss"]))
        model = keras.models.load_model(ckpt)  # best epoch by validation loss
        scores = model.predict(xs[te], verbose=0).ravel()
        m = compute_metrics(y[te], (scores >= 0.5).astype(int), scores)
        m.update({"run": r, "seed": a.seed + r, "best_val_loss": val_loss,
                  "epochs_run": len(hist.history["loss"])})
        runs.append(m)
        print(f"run {r}: val_loss={val_loss:.4f}  test: {format_metrics(m)}")
        if val_loss < best_val:
            best_val, best_run = val_loss, r
            best_scores, best_hist = scores, hist.history

    shutil.copy(f"checkpoints/hiknet_run{best_run}.keras", "checkpoints/hiknet_best.keras")
    np.save("results/cnn_test_scores.npy", best_scores)

    summary = {k: {"mean": float(np.mean([m[k] for m in runs])),
                   "std": float(np.std([m[k] for m in runs]))} for k in METRIC_KEYS}
    out = {"model": "HIKNet", "n_runs": a.runs, "selected_run": best_run,
           "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
           "summary": summary, "runs": runs}
    with open("results/cnn_metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(best_hist["loss"], label="train")
    ax.plot(best_hist["val_loss"], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/training_curves.png", dpi=150)

    print("\nmean over %d run(s) on the test split (n=%d):" % (a.runs, len(te)))
    for k in METRIC_KEYS:
        print(f"  {k:12s} {summary[k]['mean']:.3f} +/- {summary[k]['std']:.3f}")


if __name__ == "__main__":
    main()
