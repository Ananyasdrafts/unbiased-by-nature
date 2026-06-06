"""Per-instance bias detectors that DCA is benchmarked against, each returning a
score where higher means more likely biased.

  - raw_gap:          the per-instance danger signal with no aggregation. The "does
                      DCA's neighbourhood aggregation add anything?" baseline.
  - isolation_forest: a generic anomaly detector over the same fairness signals.
  - slice_finder:     axis-aligned slices ranked by within-slice group disparity (a
                      simplified Slice Finder, the subgroup-auditor competitor).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from unbiased.signals.fairness import Signals


def raw_gap_score(signals: Signals) -> np.ndarray:
    """The danger signal used directly, no DCA aggregation."""
    return np.asarray(signals.danger, dtype=float)


def isolation_forest_score(signal_matrix: np.ndarray, seed: int = 0) -> np.ndarray:
    """Isolation-forest anomaly score over the fairness signals (higher = more anomalous)."""
    X = np.asarray(signal_matrix, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    model = IsolationForest(random_state=seed).fit(X)
    return -model.score_samples(X)


def _bin(X: np.ndarray, n_bins: int) -> np.ndarray:
    binned = np.zeros(X.shape, dtype=int)
    for j in range(X.shape[1]):
        edges = np.quantile(X[:, j], np.linspace(0, 1, n_bins + 1)[1:-1])
        binned[:, j] = np.digitize(X[:, j], edges)
    return binned


def slice_finder_score(
    X: np.ndarray,
    group: np.ndarray,
    scores: np.ndarray,
    max_order: int = 2,
    n_bins: int = 4,
    min_size: int = 20,
    min_per_group: int = 5,
    top_k: int = 10,
    max_pair_features: int = 8,
) -> np.ndarray:
    """Score each instance by the within-slice group disparity of the worst axis-aligned
    slice it belongs to. Slices are single features and (capped) pairs of binned features."""
    X = np.asarray(X, dtype=float)
    group = np.asarray(group)
    scores = np.asarray(scores, dtype=float)
    n, d = X.shape
    binned = _bin(X, n_bins)

    singles = []
    for j in range(d):
        for b in np.unique(binned[:, j]):
            m = binned[:, j] == b
            if m.sum() >= min_size:
                singles.append((j, m))

    slices = [m for _, m in singles]
    if max_order >= 2:
        pair_pool = [s for s in singles if s[0] < max_pair_features]
        for i in range(len(pair_pool)):
            for j in range(i + 1, len(pair_pool)):
                if pair_pool[i][0] == pair_pool[j][0]:
                    continue
                m = pair_pool[i][1] & pair_pool[j][1]
                if m.sum() >= min_size:
                    slices.append(m)

    scored = []
    for m in slices:
        priv = scores[m & (group == 1)]
        unp = scores[m & (group == 0)]
        if len(priv) < min_per_group or len(unp) < min_per_group:
            continue
        scored.append((abs(priv.mean() - unp.mean()), m))

    scored.sort(key=lambda t: -t[0])
    out = np.zeros(n)
    for disparity, m in scored[:top_k]:
        out[m] = np.maximum(out[m], disparity)
    return out
