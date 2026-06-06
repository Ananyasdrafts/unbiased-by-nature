"""Multi-signal fairness mapping: DCA's real strength is fusing several weak, noisy
danger signals into a robust context. We compute four complementary local signals plus
a PAMP, fuse them, and build the DCA neighbourhood in a bias-relevant feature space so a
localized pocket is not diluted by irrelevant dimensions.

Danger signals (each rank-normalised to [0, 1]):
  - gap:        individual cross-group treatment gap (this instance vs similar other-group)
  - disparity:  local demographic disparity in the instance's neighbourhood
  - tpr_gap:    local true-positive-rate gap in the neighbourhood (needs labels)
  - residual:   deviation from a fair reference model trained without the protected attr
PAMP:
  - cf_flip:    counterfactual change when the protected attribute is flipped
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unbiased.signals.neighbors import (
    cross_group_neighbors,
    neighbors_all,
    within_group_neighbors,
)


@dataclass
class MultiSignals:
    danger: np.ndarray
    safe: np.ndarray
    dca_neighbors: np.ndarray
    table: dict  # named per-instance signals (for fusion baselines and ablations)


def _rank01(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return np.zeros_like(v)
    return v.argsort().argsort() / (len(v) - 1)


def _local_disparity(scores, group, nb):
    out = np.zeros(len(scores))
    for i in range(len(scores)):
        idx = nb[i]
        g, s = group[idx], scores[idx]
        if (g == 1).any() and (g == 0).any():
            out[i] = abs(s[g == 1].mean() - s[g == 0].mean())
    return out


def _local_tpr_gap(y_pred, y, group, nb):
    out = np.zeros(len(y))
    for i in range(len(y)):
        idx = nb[i]
        pos = y[idx] == 1
        rates = [y_pred[idx][pos & (group[idx] == gg)].mean() for gg in (0, 1)
                 if (pos & (group[idx] == gg)).any()]
        if len(rates) == 2:
            out[i] = abs(rates[0] - rates[1])
    return out


def _bias_relevant_weights(X, signal):
    w = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        if X[:, j].std() > 0:
            w[j] = abs(np.corrcoef(X[:, j], signal)[0, 1])
    w = np.nan_to_num(w)
    return w if w.sum() > 0 else np.ones_like(w)


def fairness_signals_multi(
    scores, group, X, y=None, cf_scores=None, fair_scores=None, k=15, pamp_weight=1.0
) -> MultiSignals:
    scores = np.asarray(scores, dtype=float)
    group = np.asarray(group)
    X = np.asarray(X, dtype=float)

    cross = cross_group_neighbors(X, group, k)
    nb_all = neighbors_all(X, k)

    gap = scores[cross].mean(axis=1) - scores  # signed
    table = {
        "gap": _rank01(np.abs(gap)),
        "disparity": _rank01(_local_disparity(scores, group, nb_all)),
    }
    if y is not None:
        table["tpr_gap"] = _rank01(_local_tpr_gap((scores >= 0.5).astype(int), np.asarray(y),
                                                  group, nb_all))
    if fair_scores is not None:
        table["residual"] = _rank01(np.abs(scores - np.asarray(fair_scores, dtype=float)))
    pamp = _rank01(np.abs(scores - np.asarray(cf_scores, dtype=float))) if cf_scores is not None \
        else np.zeros_like(scores)
    table["cf_flip"] = pamp

    danger_stack = np.column_stack([table[k_] for k_ in table if k_ != "cf_flip"])
    fused = danger_stack.mean(axis=1)
    danger = np.clip(fused + pamp_weight * pamp, 0.0, 1.0)
    safe = np.clip(1.0 - fused, 0.0, 1.0)

    weights = _bias_relevant_weights(X, danger)
    dca_neighbors = within_group_neighbors(X * weights, group, k, include_self=True)

    return MultiSignals(danger=danger, safe=safe, dca_neighbors=dca_neighbors, table=table)


def fused_signal_matrix(ms: MultiSignals) -> np.ndarray:
    """Per-instance matrix of all signals, for isolation forest / inspection."""
    return np.column_stack(list(ms.table.values()))


def naive_fusion_score(ms: MultiSignals) -> np.ndarray:
    """Mean of all signals, per instance, with no DCA aggregation. The baseline that
    asks whether DCA's aggregation beats simply averaging the same signals."""
    return fused_signal_matrix(ms).mean(axis=1)
