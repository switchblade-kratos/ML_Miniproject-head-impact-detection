"""Synthetic mouthguard data for head impact detection.

The real dataset (527 events from the Camarillo Lab mouthguard, Fall 2017) is not
public, so this module generates a stand-in with the same layout:

    data   : (N, 199, 6)   channels = lin_acc x, y, z (g), ang_vel x, y, z (rad/s)
    labels : (N, 1)        1 = real head impact, 0 = false positive

Sampling is 1000 Hz, 50 ms before the 10 g trigger and 150 ms after (199 points),
as described in the paper. Real impacts are smooth, low frequency pulses
(roughly 15-35 Hz). False positives (biting, chewing, spitting, dropping the
mouthguard) are short, high frequency, noisy bursts.

A small fraction of events of each class is made ambiguous by mixing in a
scaled copy of the other class's behaviour, so the task is not trivially
separable. Nothing here is tuned to reach a target accuracy.

Results obtained on this data say whether the pipeline works. They are not
comparable with the numbers reported on the real data.
"""

import json
import os

import numpy as np
import scipy.io as sio

FS = 1000.0          # Hz
N_POINTS = 199       # samples per event
PRE_TRIGGER_MS = 50  # t = 0 is the 10 g trigger
T = (np.arange(N_POINTS) - PRE_TRIGGER_MS) / FS  # seconds, -0.050 ... 0.148

# Difficulty. A first version (0.08, (0.3, 0.6)) was too easy: the SVM reached 100%
# cross-validation accuracy on train + validation. These values were picked from a
# small set of candidates because the SVM's cross-validation accuracy on train +
# validation (0.935) matched the 93.7% the paper reports for its SVM on real data.
# The test split was not used to choose them.
DEFAULT_AMBIGUOUS_FRAC = 0.30   # share of events that mix in the other class
DEFAULT_MIX_RANGE = (0.5, 1.0)  # strength of the mixed-in behaviour, relative to a normal event


def _unit_vector(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _pulse(t, t_peak, width, f_ring, tau, ring_frac, phase):
    """Smooth pulse plus a damped low frequency ring that starts at onset."""
    main = np.exp(-0.5 * ((t - t_peak) / width) ** 2)
    after = np.clip(t - (t_peak - 2 * width), 0, None)
    ring = ring_frac * np.exp(-after / tau) * np.sin(2 * np.pi * f_ring * after + phase)
    ring[t < (t_peak - 2 * width)] = 0.0
    return main + ring


def _burst(t, t0, freq, tau, phase):
    """Damped high frequency oscillation starting at t0."""
    s = np.clip(t - t0, 0, None)
    out = np.exp(-s / tau) * np.sin(2 * np.pi * freq * s + phase)
    out[t < t0] = 0.0
    return out


def _real_impact_lin(rng, t):
    d = _unit_vector(rng)
    amp = rng.uniform(12, 70)
    t_peak = rng.uniform(0.004, 0.020)
    width = rng.uniform(0.006, 0.014)
    shape = _pulse(t, t_peak, width, rng.uniform(15, 35), rng.uniform(0.03, 0.06),
                   rng.uniform(0.2, 0.5), rng.uniform(0, np.pi))
    out = np.empty((len(t), 3))
    for i in range(3):
        gain = d[i] + 0.15 * rng.normal()
        jitter = rng.uniform(-0.002, 0.002)
        out[:, i] = amp * gain * np.interp(t - jitter, t, shape)
    out += rng.normal(scale=rng.uniform(0.4, 1.5), size=out.shape)
    return out


def _real_impact_ang(rng, t, lin_dir_hint):
    e = 0.6 * lin_dir_hint + 0.8 * _unit_vector(rng)
    e /= np.linalg.norm(e)
    amp = rng.uniform(4, 35)
    t_peak = rng.uniform(0.008, 0.030)
    width = rng.uniform(0.010, 0.025)
    shape = _pulse(t, t_peak, width, rng.uniform(12, 28), rng.uniform(0.03, 0.07),
                   rng.uniform(0.2, 0.5), rng.uniform(0, np.pi))
    out = np.empty((len(t), 3))
    for i in range(3):
        out[:, i] = amp * (e[i] + 0.1 * rng.normal()) * shape
    out += rng.normal(scale=rng.uniform(0.3, 0.9), size=out.shape)
    return out


def _false_positive_lin(rng, t):
    n_bursts = rng.integers(1, 4)
    out = np.zeros((len(t), 3))
    t0 = rng.uniform(-0.001, 0.002)
    for _ in range(n_bursts):
        for i in range(3):
            n_comp = rng.integers(2, 5)
            for _ in range(n_comp):
                out[:, i] += (rng.choice([-1, 1]) * rng.uniform(5, 30)
                              * _burst(t, t0, rng.uniform(60, 280),
                                       rng.uniform(0.003, 0.015), rng.uniform(0, 2 * np.pi)))
        t0 += rng.uniform(0.02, 0.06)  # chewing: repeated bursts
    out += rng.normal(scale=rng.uniform(1.5, 5.0), size=out.shape)
    return out


def _false_positive_ang(rng, t):
    out = np.zeros((len(t), 3))
    t0 = rng.uniform(-0.001, 0.003)
    for i in range(3):
        for _ in range(rng.integers(1, 4)):
            out[:, i] += (rng.choice([-1, 1]) * rng.uniform(2, 20)
                          * _burst(t, t0 + rng.uniform(0, 0.02), rng.uniform(50, 220),
                                   rng.uniform(0.004, 0.02), rng.uniform(0, 2 * np.pi)))
    out += rng.normal(scale=rng.uniform(1.0, 4.0), size=out.shape)
    return out


def make_event(rng, is_impact, ambiguous, mix_range=DEFAULT_MIX_RANGE):
    if is_impact:
        lin = _real_impact_lin(rng, T)
        hint = lin.mean(axis=0)
        hint = hint / (np.linalg.norm(hint) + 1e-9)
        ang = _real_impact_ang(rng, T, hint)
        if ambiguous:
            k = rng.uniform(*mix_range)
            lin = lin + k * _false_positive_lin(rng, T)
            ang = ang + k * _false_positive_ang(rng, T)
    else:
        lin = _false_positive_lin(rng, T)
        ang = _false_positive_ang(rng, T)
        if ambiguous:
            k = rng.uniform(*mix_range)
            lin = lin + k * _real_impact_lin(rng, T)
            ang = ang + k * _real_impact_ang(rng, T, _unit_vector(rng))
    return np.concatenate([lin, ang], axis=1)  # (199, 6)


def generate(n_impact=264, n_false=263, ambiguous_frac=DEFAULT_AMBIGUOUS_FRAC,
             mix_range=DEFAULT_MIX_RANGE, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.array([1] * n_impact + [0] * n_false)
    rng.shuffle(labels)
    data = np.empty((len(labels), N_POINTS, 6))
    for k, y in enumerate(labels):
        data[k] = make_event(rng, bool(y), rng.random() < ambiguous_frac, mix_range)
    return data, labels.reshape(-1, 1).astype(float)


def save(data, labels, out_dir="data", seed=42, ambiguous_frac=DEFAULT_AMBIGUOUS_FRAC,
         mix_range=DEFAULT_MIX_RANGE):
    """Write data.mat / labels.mat using the same keys as the original code."""
    os.makedirs(out_dir, exist_ok=True)
    sio.savemat(os.path.join(out_dir, "data.mat"), {"data": data})
    sio.savemat(os.path.join(out_dir, "labels.mat"), {"label_impact_noimpact": labels})
    meta = {"synthetic": True, "n": int(len(labels)), "n_impact": int(labels.sum()),
            "fs_hz": FS, "n_points": N_POINTS, "seed": seed,
            "ambiguous_frac": ambiguous_frac, "mix_range": list(mix_range),
            "channels": ["lin_acc_x", "lin_acc_y", "lin_acc_z",
                         "ang_vel_x", "ang_vel_y", "ang_vel_z"]}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    # a split saved for older data no longer matches, so drop it
    stale = os.path.join(os.path.dirname(os.path.abspath(out_dir)), "results", "split.json")
    if os.path.exists(stale):
        os.remove(stale)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ambiguous-frac", type=float, default=DEFAULT_AMBIGUOUS_FRAC)
    p.add_argument("--mix-low", type=float, default=DEFAULT_MIX_RANGE[0])
    p.add_argument("--mix-high", type=float, default=DEFAULT_MIX_RANGE[1])
    a = p.parse_args()
    mix = (a.mix_low, a.mix_high)
    X, y = generate(ambiguous_frac=a.ambiguous_frac, mix_range=mix, seed=a.seed)
    save(X, y, a.out, a.seed, a.ambiguous_frac, mix)
    print(f"saved {X.shape} data, {int(y.sum())} impacts / {int((1 - y).sum())} false positives -> {a.out}/")
