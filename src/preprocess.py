"""Preprocessing shared by training and the demo (NumPy only)."""

import numpy as np


def standardize(x, eps=1e-8):
    """Per sample, per channel: subtract the mean and divide by the standard
    deviation over time, as described in the paper.

    x: (N, 199, 6) or (199, 6). Returns float32 of the same shape.
    """
    x = np.asarray(x, dtype=np.float64)
    axis = -2  # time axis
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    return ((x - mean) / (std + eps)).astype(np.float32)
