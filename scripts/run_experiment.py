"""Run the DCA-vs-baselines bias-recovery experiment and save the results.

    python scripts/run_experiment.py
"""

from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from unbiased.data import load_adult, load_compas, load_german_credit, load_taiwan_credit
from unbiased.eval import run_experiment
from unbiased.eval.experiment import DETECTORS

LOADERS = {
    "adult": load_adult,
    "compas": load_compas,
    "taiwan_credit": load_taiwan_credit,
    "german_credit": load_german_credit,
}


def _delta(df: pd.DataFrame, other: str) -> None:
    for (regime, pocket), sub in df.groupby(["regime", "pocket"]):
        d = sub["dca"].to_numpy() - sub[other].to_numpy()
        try:
            p = wilcoxon(sub["dca"], sub[other]).pvalue
        except ValueError:
            p = float("nan")
        print(f"  {regime:>5} {pocket:>4}: DCA - {other} = {d.mean():+.3f} "
              f"| DCA wins {int((d > 0).sum())}/{len(d)} | p={p:.3f}")


def main() -> None:
    df = run_experiment(LOADERS, pocket_kinds=("axis", "ball"),
                        regimes=("clean", "weak"), seeds=range(5))
    out = Path("docs/experiment_results.csv")
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)

    pd.set_option("display.width", 130)
    print("\n=== mean AUC by regime x pocket ===")
    print(df.groupby(["regime", "pocket"])[DETECTORS].mean().round(3).to_string())
    print("\n=== DCA vs naive_fusion (does aggregation beat just averaging the signals?) ===")
    _delta(df, "naive_fusion")
    print("\n=== DCA vs slice_finder ===")
    _delta(df, "slice_finder")


if __name__ == "__main__":
    main()
