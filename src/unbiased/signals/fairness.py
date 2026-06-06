"""Turn a model's behaviour into DCA danger / safe signals, locally and per instance.

Danger = local cross-group treatment gap: how differently this instance is scored
compared with feature-similar people from the other group (an individual / counterfactual
notion of unfairness), optionally boosted by a counterfactual flip (PAMP: does the score
depend on the protected attribute itself).

Safe = local evidence of fairness, the complement of that gap. DCA's value is not in
these per-instance numbers (a threshold could read those); it is in aggregating them over
each instance's same-group local neighbourhood, so a consistent pocket of unfairness
flags while a single noisy instance does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unbiased.signals.neighbors import cross_group_neighbors, within_group_neighbors


@dataclass
class Signals:
    danger: np.ndarray  # (n,) in [0, 1]
    safe: np.ndarray  # (n,) in [0, 1]
    dca_neighbors: np.ndarray  # (n, k+1) same-group neighbourhood for DCA aggregation
    components: dict  # raw pieces, for ablations and inspection


def fairness_signals(
    scores: np.ndarray,
    group: np.ndarray,
    X: np.ndarray,
    cf_scores: np.ndarray | None = None,
    k: int = 10,
    pamp_weight: float = 1.0,
) -> Signals:
    """Compute danger/safe signals and the DCA neighbourhood.

    scores: (n,) model probability of the positive outcome.
    group: (n,) protected attribute.
    X: (n, d) features for distances, protected attribute excluded and scaled.
    cf_scores: (n,) optional model scores with the protected attribute flipped.
    """
    scores = np.asarray(scores, dtype=float)
    group = np.asarray(group)
    X = np.asarray(X, dtype=float)

    cross = cross_group_neighbors(X, group, k)
    # signed gap: > 0 means this instance is scored lower than similar other-group people
    gap = scores[cross].mean(axis=1) - scores
    danger_gap = np.abs(gap)

    if cf_scores is not None:
        cf_change = np.abs(scores - np.asarray(cf_scores, dtype=float))
    else:
        cf_change = np.zeros_like(scores)

    danger = np.clip(danger_gap + pamp_weight * cf_change, 0.0, 1.0)
    safe = np.clip(1.0 - danger_gap, 0.0, 1.0)
    dca_neighbors = within_group_neighbors(X, group, k, include_self=True)

    return Signals(
        danger=danger,
        safe=safe,
        dca_neighbors=dca_neighbors,
        components={
            "treatment_gap": gap,  # signed; who is harmed and how much
            "danger_gap": danger_gap,
            "cf_change": cf_change,
        },
    )
