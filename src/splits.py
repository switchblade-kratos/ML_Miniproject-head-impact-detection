"""One fixed stratified train / validation / test split shared by every model.

70% train, 15% validation (used for early stopping), 15% test (touched only for
the final numbers). The indices are saved so the SVM and the CNN see exactly
the same test samples.
"""

import json
import os

import numpy as np
from sklearn.model_selection import train_test_split


def make_split(labels, seed=42, path="results/split.json"):
    y = np.asarray(labels).ravel().astype(int)
    idx = np.arange(len(y))
    train_idx, rest_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.50, stratify=y[rest_idx], random_state=seed)
    split = {"train": sorted(train_idx.tolist()), "val": sorted(val_idx.tolist()),
             "test": sorted(test_idx.tolist()), "seed": seed}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f)
    return split


def load_split(labels, seed=42, path="results/split.json"):
    if os.path.exists(path):
        with open(path) as f:
            split = json.load(f)
        if split.get("seed") == seed and sum(len(split[k]) for k in ("train", "val", "test")) == len(labels):
            return split
    return make_split(labels, seed, path)
