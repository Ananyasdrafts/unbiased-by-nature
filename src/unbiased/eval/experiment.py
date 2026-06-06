"""The core experiment: inject a known bias pocket, train a model on it, run every
detector, and score how well each recovers the injected region (AUC).

Pocket shapes: axis-aligned (a box, Slice Finder's home turf) and a ball (non-axis).
Regimes: clean (full label flip) and weak (partial flip + label noise, where the
per-instance signal is unreliable and DCA's aggregation should help if it ever does).

DCA here uses the multi-signal fusion and a bias-relevant neighbourhood space. The
naive_fusion baseline averages the same signals with no aggregation, isolating whether
DCA's aggregation adds anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from unbiased.baselines import isolation_forest_score, slice_finder_score
from unbiased.data import FairnessDataset
from unbiased.dca import DeterministicDCA
from unbiased.inject import ball_pocket, box_pocket, inject_label_bias
from unbiased.signals import (
    fairness_signals_multi,
    fused_signal_matrix,
    naive_fusion_score,
)

DETECTORS = ["dca", "naive_fusion", "raw_gap", "isolation_forest", "slice_finder"]


def run_once(
    ds: FairnessDataset,
    pocket_kind: str,
    seed: int,
    weak: bool = False,
    max_n: int = 4000,
    k: int = 15,
) -> dict:
    """One trial: AUC at recovering the injected pocket for each detector."""
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

    flip = 0.5 if weak else 1.0
    y_bias, _ = inject_label_bias(y, group, pocket, target_group=0, flip_fraction=flip, seed=seed)
    if weak:  # add label noise so the per-instance signal is unreliable
        noise = rng.random(len(y_bias)) < 0.10
        y_bias = np.where(noise, 1 - y_bias, y_bias)

    feat = np.column_stack([X, group])
    clf = RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=-1).fit(feat, y_bias)
    scores = clf.predict_proba(feat)[:, 1]
    cf = clf.predict_proba(np.column_stack([X, 1 - group]))[:, 1]
    # a fair reference model that never sees the protected attribute
    fair = RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=-1).fit(X, y_bias)
    fair_scores = fair.predict_proba(X)[:, 1]

    ms = fairness_signals_multi(scores, group, dist, y=y, cf_scores=cf, fair_scores=fair_scores, k=k)
    dca = DeterministicDCA(seed=seed, safe_weight=1.0).score(ms.danger, ms.safe, ms.dca_neighbors)

    gt = pocket.astype(int)
    preds = {
        "dca": dca.mcav,
        "naive_fusion": naive_fusion_score(ms),
        "raw_gap": ms.table["gap"],
        "isolation_forest": isolation_forest_score(fused_signal_matrix(ms), seed),
        "slice_finder": slice_finder_score(dist, group, scores),
    }
    return {name: roc_auc_score(gt, p) for name, p in preds.items()}


def run_experiment(
    loaders: dict,
    pocket_kinds=("axis", "ball"),
    regimes=("clean", "weak"),
    seeds=range(5),
    max_n: int = 4000,
) -> pd.DataFrame:
    """Run every dataset x pocket x regime x seed and return a tidy results frame."""
    rows = []
    for ds_name, loader in loaders.items():
        ds = loader()
        for pocket_kind in pocket_kinds:
            for regime in regimes:
                for seed in seeds:
                    aucs = run_once(ds, pocket_kind, seed, weak=(regime == "weak"), max_n=max_n)
                    rows.append(
                        {"dataset": ds_name, "pocket": pocket_kind, "regime": regime,
                         "seed": seed, **aucs}
                    )
    return pd.DataFrame(rows)
