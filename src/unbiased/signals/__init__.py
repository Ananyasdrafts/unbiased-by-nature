"""Map local fairness quantities to DCA danger / safe signals."""

from unbiased.signals.fairness import Signals, fairness_signals
from unbiased.signals.multi import (
    MultiSignals,
    fairness_signals_multi,
    fused_signal_matrix,
    naive_fusion_score,
)
from unbiased.signals.neighbors import (
    cross_group_neighbors,
    neighbors_all,
    within_group_neighbors,
)

__all__ = [
    "Signals",
    "fairness_signals",
    "MultiSignals",
    "fairness_signals_multi",
    "fused_signal_matrix",
    "naive_fusion_score",
    "cross_group_neighbors",
    "within_group_neighbors",
    "neighbors_all",
]
