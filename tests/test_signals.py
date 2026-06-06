"""Phase-2 pipeline test: a known local-bias pocket should light up the signals and
get flagged by DCA. Runs in CI."""

import numpy as np
from sklearn.metrics import roc_auc_score

from unbiased.dca import DeterministicDCA
from unbiased.signals import fairness_signals


def _biased_pocket(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, 2))
    group = rng.integers(0, 2, n)
    pocket = (X[:, 0] > 0.6) & (X[:, 1] > 0.6)
    score = np.full(n, 0.5)
    score[pocket] = 0.9  # a high-scoring region...
    harmed = pocket & (group == 0)
    score[harmed] = 0.05  # ...except group 0 here is pushed down (injected local bias)
    score = np.clip(score + rng.normal(0, 0.02, n), 0, 1)
    return X, group, score, pocket, harmed


def test_signals_light_up_in_the_pocket():
    X, group, score, pocket, _ = _biased_pocket()
    sig = fairness_signals(score, group, X, k=8)
    assert sig.danger[pocket].mean() > 0.6
    assert sig.danger[pocket].mean() > sig.danger[~pocket].mean()


def test_signed_gap_says_who_is_harmed():
    X, group, score, pocket, harmed = _biased_pocket()
    sig = fairness_signals(score, group, X, k=8)
    gap = sig.components["treatment_gap"]
    advantaged = pocket & (group == 1)
    assert gap[harmed].mean() > 0  # group 0 scored below similar group-1 people
    assert gap[advantaged].mean() < 0  # group 1 scored above similar group-0 people


def test_dca_flags_the_pocket():
    X, group, score, pocket, _ = _biased_pocket()
    sig = fairness_signals(score, group, X, k=8)
    res = DeterministicDCA(seed=0).score(sig.danger, sig.safe, sig.dca_neighbors)
    assert res.mcav[pocket].mean() > res.mcav[~pocket].mean()
    assert roc_auc_score(pocket.astype(int), res.mcav) > 0.8
