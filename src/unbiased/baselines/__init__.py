"""Baselines: global fairness metrics (context) and per-instance bias detectors that
DCA is benchmarked against."""

from unbiased.baselines.detectors import (
    isolation_forest_score,
    raw_gap_score,
    slice_finder_score,
)
from unbiased.baselines.metrics import (
    demographic_parity_difference,
    equal_opportunity_difference,
    equalized_odds_difference,
)

__all__ = [
    "demographic_parity_difference",
    "equal_opportunity_difference",
    "equalized_odds_difference",
    "raw_gap_score",
    "isolation_forest_score",
    "slice_finder_score",
]
