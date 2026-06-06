"""Baseline tests. Pure numpy + sklearn, runs in CI."""

import numpy as np
from sklearn.metrics import roc_auc_score

from unbiased.baselines import (
    demographic_parity_difference,
    equal_opportunity_difference,
    equalized_odds_difference,
    isolation_forest_score,
    slice_finder_score,
)


def test_global_metrics_on_a_known_case():
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    group = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # privileged always 1, unprivileged always 0
    assert demographic_parity_difference(y_pred, group) == 1.0
    assert equal_opportunity_difference(y_true, y_pred, group) == 1.0
    assert equalized_odds_difference(y_true, y_pred, group) == 1.0


def _pocket_scores(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, 2))
    group = rng.integers(0, 2, n)
    pocket = (X[:, 0] > 0.6) & (X[:, 1] > 0.6)
    scores = np.full(n, 0.5)
    scores[pocket] = 0.9
    scores[pocket & (group == 0)] = 0.05
    scores = np.clip(scores + rng.normal(0, 0.02, n), 0, 1)
    return X, group, scores, pocket


def test_isolation_forest_flags_anomalous_signals():
    rng = np.random.default_rng(0)
    danger = np.concatenate([rng.uniform(0, 0.2, 200), rng.uniform(0.8, 1, 30)])
    safe = 1 - danger
    label = np.concatenate([np.zeros(200), np.ones(30)])
    score = isolation_forest_score(np.column_stack([danger, safe]))
    assert roc_auc_score(label, score) > 0.8


def test_slice_finder_recovers_axis_aligned_pocket():
    X, group, scores, pocket = _pocket_scores()
    s = slice_finder_score(X, group, scores, min_size=15)
    assert roc_auc_score(pocket.astype(int), s) > 0.8
