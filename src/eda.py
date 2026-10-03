"""Plots of real impacts vs false positives in the time and frequency domain.

Writes figures/example_traces.png (one example of each class, like Figure 1 of the
paper) and figures/mean_spectrum.png (average spectrum of every sample per class).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from src.synthetic_data import FS, T


def load(data_dir="data"):
    x = sio.loadmat(os.path.join(data_dir, "data.mat"))["data"]
    y = sio.loadmat(os.path.join(data_dir, "labels.mat"))["label_impact_noimpact"].ravel()
    return x, y


def spectrum(sig, fs=FS):
    """Single sided amplitude spectrum, same scaling as fft_freq.m."""
    n = len(sig)
    y = np.fft.rfft(sig) / n
    y[1:-1] *= 2
    return np.fft.rfftfreq(n, 1 / fs), np.abs(y)


def example_traces(x, y, out_dir="figures", idx_true=0, idx_false=0):
    true_ex = x[np.where(y == 1)[0][idx_true]]
    false_ex = x[np.where(y == 0)[0][idx_false]]
    rows = [("Angular velocity (rad/s)", slice(3, 6)), ("Linear acceleration (g)", slice(0, 3))]
    names = ["x", "y", "z"]
    titles = ["True impact: time", "False positive: time", "True impact: frequency", "False positive: frequency"]

    fig, axes = plt.subplots(2, 4, figsize=(14, 5.5))
    for r, (label, sl) in enumerate(rows):
        for c, ex in enumerate([true_ex, false_ex]):
            ax = axes[r, c]
            for k in range(3):
                ax.plot(T, ex[:, sl][:, k], lw=1, label=names[k])
            ax.plot(T, np.linalg.norm(ex[:, sl], axis=1), "k", lw=1.5, label="mag")
            ax.set_xlim(-0.01, 0.1)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(label)
            if r == 0:
                ax.set_title(titles[c])
        for c, ex in enumerate([true_ex, false_ex]):
            ax = axes[r, c + 2]
            for k in range(3):
                f, a = spectrum(ex[:, sl][:, k])
                ax.plot(f, a, lw=1)
            f, a = spectrum(np.linalg.norm(ex[:, sl], axis=1))
            ax.plot(f, a, "k", lw=1.5)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Amplitude")
            if r == 0:
                ax.set_title(titles[c + 2])
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "example_traces.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def mean_spectrum(x, y, out_dir="figures"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (label, sl) in zip(axes, [("Linear acceleration magnitude", slice(0, 3)),
                                      ("Angular velocity magnitude", slice(3, 6))]):
        for cls, name, col in [(1, "true impact", "tab:blue"), (0, "false positive", "tab:red")]:
            specs = []
            for ex in x[y == cls]:
                f, a = spectrum(np.linalg.norm(ex[:, sl], axis=1))
                specs.append(a)
            specs = np.array(specs)
            ax.plot(f, specs.mean(axis=0), color=col, label=name)
            ax.fill_between(f, np.percentile(specs, 25, axis=0), np.percentile(specs, 75, axis=0),
                            color=col, alpha=0.2)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title(label)
        ax.set_xlim(0, FS / 2)
        ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "mean_spectrum.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    X, Y = load()
    print(example_traces(X, Y))
    print(mean_spectrum(X, Y))
