"""The core experiment: inject a known bias pocket, train a model on it, run every
detector, and score how well each recovers the injected region (AUC).

Two pocket shapes per dataset: axis-aligned (a box, Slice Finder's home turf) and a
ball (non-axis-aligned, where box slices cannot fit cleanly). The honest question is
whether DCA's neighbourhood aggregation wins on the ball.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from unbiased.baselines import isolation_forest_score, raw_gap_score, slice_finder_score
from unbiased.data import FairnessDataset
from unbiased.dca import DeterministicDCA
from unbiased.inject import ball_pocket, box_pocket, inject_label_bias
from unbiased.signals import fairness_signals

DETECTORS = ["dca", "raw_gap", "isolation_forest", "slice_finder"]


def run_once(
    ds: FairnessDataset,
    pocket_kind: str,
    seed: int,
    max_n: int = 4000,
    k: int = 15,
) -> dict:
    """One trial: returns AUC at recovering the injected pocket for each detector."""
    rng = np.random.default_rng(seed)
    n = len(ds.y)
    idx = rng.choice(n, max_n, replace=False) if n > max_n else np.arange(n)
    X, dist, y, group = ds.X[idx], ds.distance_X[idx], ds.y[idx], ds.group[idx]

    if pocket_kind == "axis":
        pocket = box_pocket(dist, dims=(0, 1), quantile=0.6)
    elif pocket_kind == "ball":
        pocket = ball_pocket(dist, dims=(0, 1, 2, 3), fraction=0.16, seed=seed)
    else:
        raise ValueError(pocket_kind)

    y_bias, _ = inject_label_bias(y, group, pocket, target_group=0, flip_fraction=1.0, seed=seed)

    feat = np.column_stack([X, group])
    clf = RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=-1).fit(feat, y_bias)
    scores = clf.predict_proba(feat)[:, 1]
    cf = clf.predict_proba(np.column_stack([X, 1 - group]))[:, 1]

    sig = fairness_signals(scores, group, dist, cf_scores=cf, k=k)
    dca = DeterministicDCA(seed=seed, safe_weight=1.0).score(sig.danger, sig.safe, sig.dca_neighbors)

    gt = pocket.astype(int)
    preds = {
        "dca": dca.mcav,
        "raw_gap": raw_gap_score(sig),
        "isolation_forest": isolation_forest_score(np.column_stack([sig.danger, sig.safe]), seed),
        "slice_finder": slice_finder_score(dist, group, scores),
    }
    return {name: roc_auc_score(gt, p) for name, p in preds.items()}


def run_experiment(
    loaders: dict,
    pocket_kinds=("axis", "ball"),
    seeds=range(5),
    max_n: int = 4000,
) -> pd.DataFrame:
    """Run every dataset x pocket x seed and return a tidy results frame."""
    rows = []
    for ds_name, loader in loaders.items():
        ds = loader()
        for pocket_kind in pocket_kinds:
            for seed in seeds:
                aucs = run_once(ds, pocket_kind, seed, max_n=max_n)
                rows.append({"dataset": ds_name, "pocket": pocket_kind, "seed": seed, **aucs})
    return pd.DataFrame(rows)
