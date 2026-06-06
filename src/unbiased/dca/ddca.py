"""Deterministic Dendritic Cell Algorithm (dDCA), adapted for tabular data.

Standard DCA assumes danger signals are concentrated in time (anomalies arrive in
bursts). Tabular fairness data is not a time series, so random sampling makes every
cell's context average out to "normal" and nothing is ever flagged. The adaptation
here is to give each cell a *local* context: a cell samples an instance together with
its feature-space neighbourhood. A biased pocket then looks locally dangerous and
matures the cell, while a fair region keeps it semi-mature. This both fixes the
degeneracy and is the whole point: DCA becomes a detector of local pockets of bias,
and its aggregation over neighbourhoods denoises single odd instances.

Per instance the two signals are danger (D) and safe (S). An exposed cell accumulates

    csm = sum over the neighbourhood of (S + D)     # drives migration
    k   = sum over the neighbourhood of (D - 2*S)    # context: k > 0 == anomalous

When csm passes the cell's randomised migration threshold it migrates, presents every
instance it sampled with context sign(k), and resets. An instance's MCAV is the
fraction of its presentations that were anomalous; flagged if MCAV > threshold (0.69).

Refs: Greensmith and Aickelin, "The Deterministic Dendritic Cell Algorithm"
(ICARIS 2008); arXiv:1006.1512, arXiv:1003.0319.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ANOMALOUS_MCAV = 0.69


@dataclass
class DCAResult:
    mcav: np.ndarray  # (n,) anomaly score in [0, 1]
    anomalous: np.ndarray  # bool mask, mcav > threshold
    threshold: float
    presentations: np.ndarray  # (n,) how many times each instance was presented


class DeterministicDCA:
    """Local-neighbourhood deterministic DCA over per-instance danger/safe signals."""

    def __init__(
        self,
        n_cells: int = 100,
        lifespan_range: tuple[float, float] = (5.0, 15.0),
        n_passes: int = 3,
        anomaly_threshold: float = ANOMALOUS_MCAV,
        seed: int = 0,
    ) -> None:
        self.n_cells = n_cells
        self.lifespan_range = lifespan_range
        self.n_passes = n_passes
        self.anomaly_threshold = anomaly_threshold
        self.seed = seed

    def score(
        self,
        danger: np.ndarray,
        safe: np.ndarray,
        neighbors: np.ndarray | None = None,
    ) -> DCAResult:
        """Score each instance.

        danger, safe: (n,) non-negative arrays.
        neighbors: (n, m) int array where row i lists instance i and its neighbourhood.
            If None, each instance is its own neighbourhood (no locality).
        """
        danger = np.asarray(danger, dtype=float)
        safe = np.asarray(safe, dtype=float)
        if danger.shape != safe.shape or danger.ndim != 1:
            raise ValueError("danger and safe must be 1-D arrays of equal length")
        n = len(danger)
        if neighbors is None:
            neighbors = np.arange(n).reshape(n, 1)
        neighbors = np.asarray(neighbors)

        rng = np.random.default_rng(self.seed)
        csm = np.zeros(self.n_cells)
        kval = np.zeros(self.n_cells)
        threshold = rng.uniform(*self.lifespan_range, size=self.n_cells)
        sampled: list[list[int]] = [[] for _ in range(self.n_cells)]

        mature = np.zeros(n)
        total = np.zeros(n)

        def migrate(c: int) -> None:
            if sampled[c]:
                idx = sampled[c]
                total[idx] += 1
                if kval[c] > 0:
                    mature[idx] += 1
            csm[c] = 0.0
            kval[c] = 0.0
            threshold[c] = rng.uniform(*self.lifespan_range)
            sampled[c] = []

        seeds = np.concatenate([rng.permutation(n) for _ in range(self.n_passes)])
        for j, s in enumerate(seeds):
            c = j % self.n_cells
            nb = neighbors[s]
            csm[c] += float((safe[nb] + danger[nb]).sum())
            kval[c] += float((danger[nb] - 2.0 * safe[nb]).sum())
            sampled[c].extend(int(x) for x in nb)
            if csm[c] >= threshold[c]:
                migrate(c)

        for c in range(self.n_cells):
            migrate(c)

        mcav = np.divide(mature, total, out=np.zeros(n), where=total > 0)
        return DCAResult(
            mcav=mcav,
            anomalous=mcav > self.anomaly_threshold,
            threshold=self.anomaly_threshold,
            presentations=total.astype(int),
        )
