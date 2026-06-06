"""Run the bias-recovery experiment (DCA vs baselines) and save the results.

    python scripts/run_experiment.py
"""

from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from unbiased.data import load_adult, load_compas, load_german_credit, load_taiwan_credit
from unbiased.eval import run_experiment

LOADERS = {
    "adult": load_adult,
    "compas": load_compas,
    "taiwan_credit": load_taiwan_credit,
    "german_credit": load_german_credit,
}
DETECTORS = ["dca", "raw_gap", "isolation_forest", "slice_finder"]


def main() -> None:
    df = run_experiment(LOADERS, pocket_kinds=("axis", "ball"), seeds=range(5))
    out = Path("docs/experiment_results.csv")
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)

    pd.set_option("display.width", 120)
    print("\n=== mean AUC by pocket type (across datasets x seeds) ===")
    print(df.groupby("pocket")[DETECTORS].mean().round(3).to_string())
    print("\n=== mean AUC by dataset x pocket ===")
    print(df.groupby(["pocket", "dataset"])[DETECTORS].mean().round(3).to_string())

    print("\n=== DCA vs Slice Finder (paired across runs) ===")
    for pk in ("axis", "ball"):
        sub = df[df.pocket == pk]
        delta = sub["dca"].to_numpy() - sub["slice_finder"].to_numpy()
        try:
            p = wilcoxon(sub["dca"], sub["slice_finder"]).pvalue
        except ValueError:
            p = float("nan")
        wins = int((delta > 0).sum())
        print(f"  {pk:>4}: DCA - Slice mean delta = {delta.mean():+.3f} | "
              f"DCA wins {wins}/{len(delta)} | wilcoxon p={p:.3f}")


if __name__ == "__main__":
    main()
