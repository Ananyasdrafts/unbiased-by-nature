"""Loaders for the standard fairness benchmarks.

Conventions (following the AIF360 / fairness literature): label 1 = favorable outcome,
group 1 = privileged. Data is fetched once and cached under data/ (gitignored).

  - Adult:   target income >50K; protected sex (privileged = Male)
  - COMPAS:  target two-year recidivism (favorable = no recid); protected race
             (privileged = Caucasian), standard ProPublica row filtering, restricted
             to Caucasian vs African-American
  - Taiwan:  target default next month (favorable = no default); protected sex
             (privileged = male)
  - German:  target good credit; protected age (privileged = age > 25)
"""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import fetch_openml

from unbiased.data.base import DATA_HOME, FairnessDataset, build_dataset

_OPENML = DATA_HOME / "openml"
_COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/master/"
    "compas-scores-two-years.csv"
)


def _find(df: pd.DataFrame, *candidates: str) -> str:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise KeyError(f"none of {candidates} found in {list(df.columns)}")


def load_adult() -> FairnessDataset:
    d = fetch_openml("adult", version=2, as_frame=True, data_home=str(_OPENML))
    df = d.data.copy()
    df["__y__"] = d.target.astype(str).str.replace(".", "", regex=False).str.strip().values
    sex = _find(df, "sex")
    group = (df[sex].astype(str).str.strip().str.lower() == "male").astype(int).to_numpy()
    return build_dataset("adult", df, "__y__", ">50K", group, sex, [sex])


def load_german_credit() -> FairnessDataset:
    d = fetch_openml("credit-g", version=1, as_frame=True, data_home=str(_OPENML))
    df = d.data.copy()
    df["__y__"] = d.target.astype(str).values
    age = _find(df, "age")
    group = (df[age].astype(float) > 25).astype(int).to_numpy()
    return build_dataset("german_credit", df, "__y__", "good", group, "age", [age])


def load_taiwan_credit() -> FairnessDataset:
    d = fetch_openml(
        "default-of-credit-card-clients", version=1, as_frame=True, data_home=str(_OPENML)
    )
    df = d.data.copy()
    df["__y__"] = d.target.astype(str).values
    sex = _find(df, "x2", "sex", "SEX")
    group = (df[sex].astype(float) == 1).astype(int).to_numpy()  # 1 = male
    return build_dataset("taiwan_credit", df, "__y__", "0", group, sex, [sex])


def load_compas() -> FairnessDataset:
    path = DATA_HOME / "compas-scores-two-years.csv"
    if not path.exists():
        DATA_HOME.mkdir(parents=True, exist_ok=True)
        pd.read_csv(_COMPAS_URL).to_csv(path, index=False)
    raw = pd.read_csv(path)
    keep = (
        (raw["days_b_screening_arrest"] <= 30)
        & (raw["days_b_screening_arrest"] >= -30)
        & (raw["is_recid"] != -1)
        & (raw["c_charge_degree"] != "O")
        & (raw["score_text"] != "N/A")
        & (raw["race"].isin(["Caucasian", "African-American"]))
    )
    raw = raw[keep]
    cols = [
        "sex", "age", "race", "juv_fel_count", "juv_misd_count",
        "juv_other_count", "priors_count", "c_charge_degree",
    ]
    df = raw[cols].copy()
    df["__y__"] = raw["two_year_recid"].astype(str).values
    group = (df["race"] == "Caucasian").astype(int).to_numpy()
    return build_dataset("compas", df, "__y__", "0", group, "race", ["race"])


_LOADERS = {
    "adult": load_adult,
    "compas": load_compas,
    "taiwan_credit": load_taiwan_credit,
    "german_credit": load_german_credit,
}


def load_dataset(name: str) -> FairnessDataset:
    if name not in _LOADERS:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(_LOADERS)}")
    return _LOADERS[name]()
