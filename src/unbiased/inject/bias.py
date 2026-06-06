"""Localized bias injection for ground-truth evaluation.

Three classic mechanisms, each confined to a feature-space pocket and targeting one
group, so the model trained on the result is unfair in a known place:

  - label bias:        historically unfavourable labels for the target group (flip
                       favourable outcomes to unfavourable inside the pocket)
  - measurement bias:  a feature recorded worse for the target group in the pocket
  - sampling bias:      the target group under-represented in the pocket

Each returns the affected region as a boolean mask (the ground truth).
"""

from __future__ import annotations

import numpy as np


def box_pocket(X: np.ndarray, dims: tuple[int, ...] = (0, 1), quantile: float = 0.6) -> np.ndarray:
    """A pocket = instances above the given quantile on every listed feature."""
    X = np.asarray(X, dtype=float)
    mask = np.ones(len(X), dtype=bool)
    for d in dims:
        mask &= X[:, d] > np.quantile(X[:, d], quantile)
    return mask


def inject_label_bias(
    y: np.ndarray,
    group: np.ndarray,
    pocket: np.ndarray,
    target_group=0,
    flip_fraction: float = 1.0,
    favorable=1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Flip favourable labels to unfavourable for the target group inside the pocket.

    Returns (biased labels, biased_region mask = pocket and target group)."""
    y = np.asarray(y).copy()
    group = np.asarray(group)
    rng = np.random.default_rng(seed)
    region = pocket & (group == target_group)
    candidates = np.where(region & (y == favorable))[0]
    n_flip = int(round(flip_fraction * len(candidates)))
    flip = rng.choice(candidates, size=n_flip, replace=False) if n_flip else np.array([], int)
    y[flip] = 1 - favorable if favorable in (0, 1) else y[flip]
    return y, region


def inject_measurement_bias(
    X: np.ndarray,
    group: np.ndarray,
    pocket: np.ndarray,
    feature: int,
    target_group=0,
    shift: float = -1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Corrupt one feature (a worse-recorded measurement) for the target group in the
    pocket. Returns (biased X, biased_region mask)."""
    X = np.asarray(X, dtype=float).copy()
    group = np.asarray(group)
    region = pocket & (group == target_group)
    X[region, feature] += shift
    return X, region


def inject_sampling_bias(
    group: np.ndarray,
    pocket: np.ndarray,
    target_group=0,
    drop_fraction: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Under-represent the target group inside the pocket by dropping a fraction of them.

    Returns (keep mask over the original rows, biased_region mask over the original rows).
    The caller subsets X/y/group by the keep mask; the region tells you where coverage
    was thinned."""
    group = np.asarray(group)
    rng = np.random.default_rng(seed)
    region = pocket & (group == target_group)
    keep = np.ones(len(group), dtype=bool)
    candidates = np.where(region)[0]
    n_drop = int(round(drop_fraction * len(candidates)))
    if n_drop:
        keep[rng.choice(candidates, size=n_drop, replace=False)] = False
    return keep, region
