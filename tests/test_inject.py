"""Injection unit tests plus the headline integration test: inject bias, train a
model on it, and check the detector recovers where the bias was put. Runs in CI."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from unbiased.dca import DeterministicDCA
from unbiased.inject import (
    box_pocket,
    inject_label_bias,
    inject_measurement_bias,
    inject_sampling_bias,
)
from unbiased.signals import fairness_signals


def test_box_pocket_is_a_region():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (1000, 3))
    pocket = box_pocket(X, dims=(0, 1), quantile=0.6)
    assert 0.1 < pocket.mean() < 0.25  # ~ (0.4)^2


def test_label_bias_flips_only_in_region():
    rng = np.random.default_rng(0)
    n = 800
    X = rng.uniform(0, 1, (n, 2))
    group = rng.integers(0, 2, n)
    y = rng.integers(0, 2, n)
    pocket = box_pocket(X, quantile=0.5)
    y_bias, region = inject_label_bias(y, group, pocket, target_group=0, seed=1)
    changed = y_bias != y
    assert region.sum() > 0
    assert changed.sum() > 0
    assert np.all(region[changed])  # every change is inside the target region
    assert np.all((group[changed] == 0) & pocket[changed])


def test_measurement_bias_only_touches_region():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (500, 3))
    group = rng.integers(0, 2, 500)
    pocket = box_pocket(X, quantile=0.5)
    Xb, region = inject_measurement_bias(X, group, pocket, feature=2, target_group=0, shift=-1.0)
    moved = ~np.isclose(Xb[:, 2], X[:, 2])
    assert np.array_equal(moved, region)


def test_sampling_bias_drops_a_fraction():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (1000, 2))
    group = rng.integers(0, 2, 1000)
    pocket = box_pocket(X, quantile=0.5)
    keep, region = inject_sampling_bias(group, pocket, target_group=0, drop_fraction=0.5, seed=1)
    dropped = ~keep
    assert np.all(region[dropped])  # only the target region is thinned
    assert abs(dropped.sum() - 0.5 * region.sum()) <= 1


def test_recovers_injected_label_bias_end_to_end():
    rng = np.random.default_rng(0)
    n = 1500
    X = rng.uniform(0, 1, (n, 4))
    group = rng.integers(0, 2, n)
    # fair ground-truth labels (depend on features, not group)
    p = 1 / (1 + np.exp(-4 * (X[:, 0] + X[:, 1] - 1)))
    y = (rng.uniform(size=n) < p).astype(int)

    pocket = box_pocket(X, dims=(0, 1), quantile=0.5)
    y_bias, region = inject_label_bias(y, group, pocket, target_group=0, flip_fraction=1.0, seed=1)

    feat = np.column_stack([X, group])  # model can see the protected attribute
    clf = RandomForestClassifier(n_estimators=150, random_state=0).fit(feat, y_bias)
    scores = clf.predict_proba(feat)[:, 1]
    cf_scores = clf.predict_proba(np.column_stack([X, 1 - group]))[:, 1]

    Xs = StandardScaler().fit_transform(X)
    sig = fairness_signals(scores, group, Xs, cf_scores=cf_scores, k=15)
    res = DeterministicDCA(seed=0, safe_weight=1.0).score(sig.danger, sig.safe, sig.dca_neighbors)

    # the detector should recover the injected-bias pocket
    assert roc_auc_score(pocket.astype(int), res.mcav) > 0.7
    # and the harmed group should carry positive signed gap
    assert sig.components["treatment_gap"][region].mean() > 0
