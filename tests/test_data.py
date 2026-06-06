"""Dataset tests. The encoding logic is tested offline; the real downloads are
opt-in (set RUN_DATA_TESTS=1) so CI stays fast and network-free."""

import os

import numpy as np
import pandas as pd
import pytest

from unbiased.data.base import build_dataset


def test_build_dataset_encodes_scales_and_excludes_protected():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "color": ["r", "g", "r", "b"],
            "sex": ["M", "F", "M", "F"],
            "target": ["yes", "no", "yes", "no"],
        }
    )
    group = (df["sex"] == "M").astype(int).to_numpy()
    ds = build_dataset("toy", df, "target", "yes", group, "sex", ["sex"])

    assert ds.y.tolist() == [1, 0, 1, 0]  # favorable == "yes"
    assert ds.group.tolist() == [1, 0, 1, 0]  # privileged == "M"
    # X = age + color(r,g,b) + sex(F,M) = 6 columns; distance drops the 2 sex columns
    assert ds.X.shape == (4, 6)
    assert ds.distance_X.shape == (4, 4)
    assert abs(float(ds.distance_X.mean())) < 1e-9  # standard-scaled


@pytest.mark.skipif(not os.environ.get("RUN_DATA_TESTS"), reason="downloads data")
def test_real_loaders_smoke():
    from unbiased.data import load_dataset

    for name in ("german_credit", "compas"):
        ds = load_dataset(name)
        assert len(ds.y) > 0
        assert set(np.unique(ds.y)) <= {0, 1}
        assert set(np.unique(ds.group)) <= {0, 1}
        assert ds.distance_X.shape[0] == len(ds.y)
