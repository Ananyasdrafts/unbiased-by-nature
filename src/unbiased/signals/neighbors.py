"""Feature-space neighbourhoods, computed on features with the protected attribute
excluded so neighbours are 'similar people' regardless of group."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _knn(cand_X: np.ndarray, cand_idx: np.ndarray, query_X: np.ndarray, k: int) -> np.ndarray:
    """k nearest candidates for each query row; returns global indices (n_query, k)."""
    k = min(k, len(cand_idx))
    nn = NearestNeighbors(n_neighbors=k).fit(cand_X)
    _, local = nn.kneighbors(query_X)
    return cand_idx[local]


def neighbors_all(X: np.ndarray, k: int = 10, include_self: bool = True) -> np.ndarray:
    """k nearest neighbours from the whole dataset (both groups), (n, k or k+1)."""
    X = np.asarray(X, dtype=float)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)
    _, nbr = nn.kneighbors(X)
    return nbr if include_self else nbr[:, 1:]


def cross_group_neighbors(X: np.ndarray, group: np.ndarray, k: int = 10) -> np.ndarray:
    """For each instance, the k nearest instances from the OTHER group (n, k)."""
    X = np.asarray(X, dtype=float)
    group = np.asarray(group)
    out = np.zeros((len(X), k), dtype=int)
    for g in np.unique(group):
        q = np.where(group == g)[0]
        opp = np.where(group != g)[0]
        out[q] = _knn(X[opp], opp, X[q], k)
    return out


def within_group_neighbors(
    X: np.ndarray, group: np.ndarray, k: int = 10, include_self: bool = True
) -> np.ndarray:
    """For each instance, the k nearest instances from the SAME group (n, k or k+1).
    With include_self, the instance itself is column 0 (used as the DCA neighbourhood)."""
    X = np.asarray(X, dtype=float)
    group = np.asarray(group)
    width = k + 1 if include_self else k
    out = np.zeros((len(X), width), dtype=int)
    for g in np.unique(group):
        q = np.where(group == g)[0]
        # query within the same group; +1 to drop self when needed
        nbr = _knn(X[q], q, X[q], min(width + 1, len(q)))
        for row, i in enumerate(q):
            same = nbr[row]
            if include_self:
                same = np.concatenate([[i], same[same != i]])[:width]
            else:
                same = same[same != i][:width]
            out[i, : len(same)] = same
            if len(same) < width:  # pad short rows with self
                out[i, len(same):] = i
    return out
