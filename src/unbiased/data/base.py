"""Common dataset structure and preprocessing.

Every loader returns a FairnessDataset with a numeric feature matrix for the model,
a binary favorable-outcome label (1 = favorable), a binary protected group
(1 = privileged), and a scaled distance matrix for neighbourhoods that excludes the
protected attribute, so "similar people" are similar regardless of group.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_HOME = Path(__file__).resolve().parents[3] / "data"


@dataclass
class FairnessDataset:
    name: str
    X: np.ndarray  # encoded numeric features for the model
    feature_names: list[str]
    y: np.ndarray  # 1 = favorable outcome
    group: np.ndarray  # 1 = privileged group
    group_name: str
    distance_X: np.ndarray  # scaled features for neighbourhoods, protected excluded

    def summary(self) -> str:
        return (
            f"{self.name}: {len(self.y)} rows, {self.X.shape[1]} features | "
            f"favorable={self.y.mean():.0%} | privileged({self.group_name})={self.group.mean():.0%}"
        )


def build_dataset(
    name: str,
    df: pd.DataFrame,
    target_col: str,
    favorable,
    group: np.ndarray,
    group_name: str,
    exclude_from_distance: list[str],
) -> FairnessDataset:
    """Encode a raw frame into a FairnessDataset.

    df: raw features + target (already filtered). group: precomputed binary privileged
    array. exclude_from_distance: original column names whose encoded versions are kept
    out of the neighbourhood distance (the protected attribute and direct proxies).
    """
    df = df.copy()
    y = (df[target_col].astype(str) == str(favorable)).astype(int).to_numpy()
    df = df.drop(columns=[target_col])

    encoded = pd.get_dummies(df, drop_first=False)
    feature_names = encoded.columns.tolist()
    X = encoded.to_numpy(dtype=float)

    drop = [
        c
        for c in encoded.columns
        if any(c == ex or c.startswith(ex + "_") for ex in exclude_from_distance)
    ]
    dist = encoded.drop(columns=drop, errors="ignore").to_numpy(dtype=float)
    distance_X = StandardScaler().fit_transform(dist)

    return FairnessDataset(
        name=name,
        X=X,
        feature_names=feature_names,
        y=y,
        group=np.asarray(group, dtype=int),
        group_name=group_name,
        distance_X=distance_X,
    )
