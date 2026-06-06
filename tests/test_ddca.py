"""DCA engine tests. Pure numpy + sklearn metric, runs in CI."""

import numpy as np
from sklearn.metrics import roc_auc_score

from unbiased.dca import DeterministicDCA


def _pocket_neighbors(label, m=10, seed=0):
    """Neighbourhoods that group like with like, simulating a local pocket:
    each instance's neighbourhood is drawn from its own class."""
    rng = np.random.default_rng(seed)
    idx_by_class = {c: np.where(label == c)[0] for c in np.unique(label)}
    rows = []
    for i, c in enumerate(label):
        pool = idx_by_class[c]
        picks = rng.choice(pool, size=min(m, len(pool)), replace=False)
        rows.append(np.unique(np.append(picks, i))[:m])
    # pad ragged rows to equal length by repeating the instance itself
    width = max(len(r) for r in rows)
    return np.array([np.pad(r, (0, width - len(r)), constant_values=i)
                     for i, r in enumerate(rows)])


def _synthetic(n_normal=200, n_anom=30, seed=0):
    rng = np.random.default_rng(seed)
    danger = np.concatenate([rng.uniform(0, 0.2, n_normal), rng.uniform(0.8, 1.0, n_anom)])
    safe = np.concatenate([rng.uniform(0.8, 1.0, n_normal), rng.uniform(0, 0.2, n_anom)])
    label = np.concatenate([np.zeros(n_normal), np.ones(n_anom)])
    return danger, safe, label


def test_flags_a_local_pocket():
    danger, safe, label = _synthetic()
    neighbors = _pocket_neighbors(label)
    res = DeterministicDCA(seed=1).score(danger, safe, neighbors)
    assert res.mcav[label == 1].mean() > res.mcav[label == 0].mean()
    assert roc_auc_score(label, res.mcav) > 0.9


def test_all_safe_flags_nothing():
    n = 100
    res = DeterministicDCA(seed=0).score(danger=np.zeros(n), safe=np.ones(n))
    assert res.anomalous.sum() == 0


def test_is_deterministic():
    danger, safe, label = _synthetic()
    neighbors = _pocket_neighbors(label)
    a = DeterministicDCA(seed=7).score(danger, safe, neighbors)
    b = DeterministicDCA(seed=7).score(danger, safe, neighbors)
    assert np.array_equal(a.mcav, b.mcav)
