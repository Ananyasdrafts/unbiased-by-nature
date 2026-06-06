"""Fairness benchmark datasets, loaded with documented preprocessing."""

from unbiased.data.base import FairnessDataset
from unbiased.data.loaders import (
    load_adult,
    load_compas,
    load_dataset,
    load_german_credit,
    load_taiwan_credit,
)

__all__ = [
    "FairnessDataset",
    "load_adult",
    "load_compas",
    "load_taiwan_credit",
    "load_german_credit",
    "load_dataset",
]
