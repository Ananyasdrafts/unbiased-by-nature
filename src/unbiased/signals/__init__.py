"""Map local fairness quantities to DCA danger / safe signals."""

from unbiased.signals.fairness import Signals, fairness_signals
from unbiased.signals.neighbors import (
    cross_group_neighbors,
    within_group_neighbors,
)

__all__ = [
    "Signals",
    "fairness_signals",
    "cross_group_neighbors",
    "within_group_neighbors",
]
