"""Inject known, controlled bias into a dataset to create ground truth.

Each injector confines the bias to a feature-space pocket and returns a mask of the
affected region, so an evaluation can ask the only honest question: did the detector
recover the bias we put in, and where.
"""

from unbiased.inject.bias import (
    box_pocket,
    inject_label_bias,
    inject_measurement_bias,
    inject_sampling_bias,
)

__all__ = [
    "box_pocket",
    "inject_label_bias",
    "inject_measurement_bias",
    "inject_sampling_bias",
]
