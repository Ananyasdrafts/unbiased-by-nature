"""Global fairness metrics. These summarise unfairness for the whole model; they do
not localise it, which is exactly the gap a local detector is meant to fill. Reported
for context alongside the detection results."""

from __future__ import annotations

import numpy as np


def _rate(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else float("nan")


def demographic_parity_difference(y_pred: np.ndarray, group: np.ndarray) -> float:
    """|P(pred=1 | privileged) - P(pred=1 | unprivileged)|."""
    y_pred = np.asarray(y_pred)
    group = np.asarray(group)
    return abs(_rate(y_pred[group == 1] == 1) - _rate(y_pred[group == 0] == 1))


def equal_opportunity_difference(
    y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray
) -> float:
    """|TPR_privileged - TPR_unprivileged| (gap in true-positive rates)."""
    y_true, y_pred, group = map(np.asarray, (y_true, y_pred, group))
    tpr = {g: _rate(y_pred[(group == g) & (y_true == 1)] == 1) for g in (0, 1)}
    return abs(tpr[1] - tpr[0])


def equalized_odds_difference(
    y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray
) -> float:
    """max gap across TPR and FPR (equalized-odds violation)."""
    y_true, y_pred, group = map(np.asarray, (y_true, y_pred, group))
    gaps = []
    for actual in (0, 1):
        rate = {g: _rate(y_pred[(group == g) & (y_true == actual)] == 1) for g in (0, 1)}
        gaps.append(abs(rate[1] - rate[0]))
    return max(gaps)
