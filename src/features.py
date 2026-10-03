"""Handcrafted features for the SVM baseline.

The prior work (Wu et al.) used time domain and power spectral density features
chosen by sequential feature selection. This is a smaller version of the same
idea: for the linear acceleration and the angular velocity separately we compute
17 features from the vector magnitude and the three axes.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

FS = 1000.0
BANDS = [(0, 10), (10, 30), (30, 80), (80, 200), (200, 500)]
GROUPS = {"lin": slice(0, 3), "ang": slice(3, 6)}


def _group_features(sig):
    """sig: (199, 3) for one sample and one group. Returns dict of features."""
    mag = np.linalg.norm(sig, axis=1)
    peak = mag.max()
    f = {}
    f["peak"] = peak
    f["t_peak_ms"] = float(np.argmax(mag))
    f["rms"] = float(np.sqrt(np.mean(mag ** 2)))
    f["kurtosis"] = float(kurtosis(mag))
    f["skew"] = float(skew(mag))
    f["roughness"] = float(np.mean(np.abs(np.diff(mag))) / (peak + 1e-9))

    peaks, _ = find_peaks(mag, height=0.5 * peak)
    f["n_peaks"] = float(len(peaks))
    all_peaks, _ = find_peaks(mag)
    g = int(np.argmax(mag))
    f["peak_order"] = float(np.searchsorted(all_peaks, g)) if len(all_peaks) else 0.0

    centred = sig - sig.mean(axis=0)
    signs = np.sign(centred)
    f["zero_cross"] = float(np.mean(np.sum(signs[1:] != signs[:-1], axis=0)))

    cov = np.cov(centred.T)
    eig = np.linalg.eigvalsh(cov)
    f["coherence"] = float(eig[-1] / (eig.sum() + 1e-9))  # 1 = motion along one axis

    m = mag - mag.mean()
    power = np.abs(np.fft.rfft(m)) ** 2
    freqs = np.fft.rfftfreq(len(m), 1 / FS)
    total = power.sum() + 1e-12
    for lo, hi in BANDS:
        sel = (freqs >= lo) & (freqs < hi) if hi < 500 else (freqs >= lo)
        f[f"band_{lo}_{hi}"] = float(power[sel].sum() / total)
    f["centroid"] = float((freqs * power).sum() / total)
    cum = np.cumsum(power) / total
    f["edge90"] = float(freqs[np.searchsorted(cum, 0.9)])
    return f


def extract_features(x):
    """x: (N, 199, 6) -> (matrix (N, F), list of feature names)."""
    rows, names = [], None
    for sample in x:
        feats = {}
        for gname, sl in GROUPS.items():
            for k, v in _group_features(sample[:, sl]).items():
                feats[f"{gname}_{k}"] = v
        if names is None:
            names = list(feats)
        rows.append([feats[n] for n in names])
    return np.array(rows, dtype=float), names
