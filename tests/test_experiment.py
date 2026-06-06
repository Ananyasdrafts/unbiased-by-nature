"""Experiment-runner test on a tiny synthetic dataset (no network). Runs in CI."""

import numpy as np
from sklearn.preprocessing import StandardScaler

from unbiased.data import FairnessDataset
from unbiased.eval import run_once


def _toy_dataset(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, 5))
    group = rng.integers(0, 2, n)
    p = 1 / (1 + np.exp(-4 * (X[:, 0] + X[:, 1] - 1)))
    y = (rng.uniform(size=n) < p).astype(int)
    dist = StandardScaler().fit_transform(X)
    return FairnessDataset(
        "toy", X, [f"f{i}" for i in range(5)], y, group, "g", dist
    )


def test_run_once_returns_aucs_for_all_detectors():
    out = run_once(_toy_dataset(), "axis", seed=0, max_n=400, k=10)
    assert set(out) == {"dca", "raw_gap", "isolation_forest", "slice_finder"}
    assert all(0.0 <= v <= 1.0 for v in out.values())
