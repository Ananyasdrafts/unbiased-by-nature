"""unbiased: detecting local pockets of algorithmic bias with an Artificial Immune
System anomaly detector (the Dendritic Cell Algorithm).

Research question: do standard fairness tools, which are mostly global averages or
axis-aligned subgroup audits, miss oddly-shaped local pockets of unfairness that a
per-instance, multi-signal, subgroup-agnostic anomaly detector can catch?

Package layout (built in phases):
    dca/        the deterministic Dendritic Cell Algorithm engine
    signals/    map local fairness quantities to danger / safe / PAMP signals
    inject/     inject known, controlled bias to create ground truth
    data/       dataset loaders (Adult, COMPAS, Taiwan credit, German credit)
    baselines/  global metrics, Slice Finder, isolation forest
    eval/       detection metrics and experiment runners
"""

__version__ = "0.0.1"
