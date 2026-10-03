"""Live demo: classify mouthguard events with the trained HIKNet.

    python -m src.demo                  # 8 random events from the held-out test split
    python -m src.demo --n 15 --seed 3  # more events, different draw
    python -m src.demo --plot           # also save a plot per event in figures/
    python -m src.demo --csv event.csv  # classify your own event: 199 rows x 6 columns
                                        # (lin_acc x,y,z then ang_vel x,y,z), no header

Needs checkpoints/hiknet_best.keras, which train_cnn.py writes.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from src.preprocess import standardize
from src.splits import load_split
from src.synthetic_data import T

LABEL = {1: "impact", 0: "false positive"}


def plot_event(event, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, (name, sl) in zip(axes, [("Linear acceleration (g)", slice(0, 3)),
                                     ("Angular velocity (rad/s)", slice(3, 6))]):
        for k, c in enumerate("xyz"):
            ax.plot(T, event[:, sl][:, k], lw=1, label=c)
        ax.plot(T, np.linalg.norm(event[:, sl], axis=1), "k", lw=1.5, label="mag")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(name)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="checkpoints/hiknet_best.keras")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--csv", default=None)
    a = p.parse_args()

    from tensorflow import keras  # imported late so --help works without TensorFlow
    model = keras.models.load_model(a.model)
    os.makedirs("figures", exist_ok=True)

    if a.csv:
        event = np.loadtxt(a.csv, delimiter=",")
        if event.shape != (199, 6):
            raise SystemExit(f"expected shape (199, 6), got {event.shape}")
        prob = float(model.predict(standardize(event)[None], verbose=0)[0, 0])
        pred = int(prob >= 0.5)
        print(f"{a.csv}: P(impact) = {prob:.3f} -> {LABEL[pred]}")
        if a.plot:
            plot_event(event, f"{os.path.basename(a.csv)}: {LABEL[pred]} ({prob:.2f})", "figures/demo_csv.png")
        return

    x = sio.loadmat("data/data.mat")["data"]
    y = sio.loadmat("data/labels.mat")["label_impact_noimpact"].ravel().astype(int)
    split = load_split(y, 42, "results/split.json")
    rng = np.random.default_rng(a.seed)
    idx = rng.choice(split["test"], size=min(a.n, len(split["test"])), replace=False)

    probs = model.predict(standardize(x[idx]), verbose=0).ravel()
    print(f"{'event':>6}  {'true label':<15} {'P(impact)':>9}  {'prediction':<15} correct")
    right = 0
    for i, pr in zip(idx, probs):
        pred = int(pr >= 0.5)
        ok = pred == y[i]
        right += ok
        print(f"{i:>6}  {LABEL[y[i]]:<15} {pr:>9.3f}  {LABEL[pred]:<15} {'yes' if ok else 'NO'}")
        if a.plot:
            plot_event(x[i], f"event {i}: true {LABEL[y[i]]}, predicted {LABEL[pred]} ({pr:.2f})",
                       f"figures/demo_event_{i}.png")
    print(f"\n{right}/{len(idx)} correct (these events were never used for training)")


if __name__ == "__main__":
    main()
