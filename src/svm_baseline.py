"""SVM baseline on handcrafted features (the approach of the prior work).

Steps: extract features -> forward sequential feature selection (5-fold CV on the
training data only) -> grid search over C and gamma -> fit -> evaluate once on the
held-out test split.

The SVM is fitted on train + validation (85% of the data) because it has no
early stopping; the CNN uses the validation part for early stopping instead.
Both are scored on the same test samples.
"""

import json
import os

import numpy as np
import scipy.io as sio
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.features import extract_features
from src.metrics import compute_metrics, format_metrics
from src.splits import load_split


def run(data_dir="data", out_dir="results", seed=42):
    x = sio.loadmat(os.path.join(data_dir, "data.mat"))["data"]
    y = sio.loadmat(os.path.join(data_dir, "labels.mat"))["label_impact_noimpact"].ravel().astype(int)
    split = load_split(y, seed, os.path.join(out_dir, "split.json"))
    fit_idx = np.array(split["train"] + split["val"])
    test_idx = np.array(split["test"])

    feats, names = extract_features(x)
    assert np.isfinite(feats).all(), "non-finite feature values"

    scaler = StandardScaler().fit(feats[fit_idx])
    z = scaler.transform(feats)

    selector = SequentialFeatureSelector(
        SVC(kernel="rbf"), n_features_to_select="auto", tol=0.002,
        direction="forward", cv=5, scoring="accuracy", n_jobs=-1)
    selector.fit(z[fit_idx], y[fit_idx])
    chosen = np.where(selector.get_support())[0]
    chosen_names = [names[i] for i in chosen]

    grid = GridSearchCV(
        Pipeline([("svc", SVC(kernel="rbf"))]),
        {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1, 1]},
        cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(z[fit_idx][:, chosen], y[fit_idx])

    model = grid.best_estimator_
    scores = model.decision_function(z[test_idx][:, chosen])
    pred = model.predict(z[test_idx][:, chosen])
    metrics = compute_metrics(y[test_idx], pred, scores)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "svm_test_scores.npy"), scores)
    result = {"model": "SVM (RBF) on handcrafted features",
              "n_features_total": len(names), "selected_features": chosen_names,
              "best_params": {k: (v if isinstance(v, str) else float(v)) for k, v in grid.best_params_.items()},
              "cv_accuracy_fit_set": float(grid.best_score_),
              "n_fit": int(len(fit_idx)), "n_test": int(len(test_idx)),
              "test_metrics": metrics}
    with open(os.path.join(out_dir, "svm_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    r = run()
    print("selected features:", r["selected_features"])
    print("best params:", r["best_params"], " cv acc: %.3f" % r["cv_accuracy_fit_set"])
    print("test (n=%d):" % r["n_test"], format_metrics(r["test_metrics"]))
