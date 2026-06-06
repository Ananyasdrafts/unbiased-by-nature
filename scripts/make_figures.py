"""Generate the README figures from the experiment results and a real bias map.

    python scripts/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from unbiased.data import load_adult  # noqa: E402
from unbiased.inject import ball_pocket, inject_label_bias  # noqa: E402
from unbiased.signals import fairness_signals_multi, naive_fusion_score  # noqa: E402

IMG = Path("docs/images")
IMG.mkdir(parents=True, exist_ok=True)
DETECTORS = ["dca", "naive_fusion", "raw_gap", "isolation_forest", "slice_finder"]
LABELS = ["DCA", "fused (simple)", "raw gap", "isolation forest", "Slice Finder"]


def benchmark_figure() -> None:
    df = pd.read_csv("docs/experiment_results.csv")
    g = df[df.regime == "clean"].groupby("pocket")[DETECTORS].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(DETECTORS))
    w = 0.38
    for i, pk in enumerate(["axis", "ball"]):
        ax.bar(x + (i - 0.5) * w, g.loc[pk].to_numpy(), w, label=f"{pk}-aligned pocket")
    ax.axhline(0.5, ls=":", c="gray", lw=1)
    ax.text(len(DETECTORS) - 1.4, 0.51, "chance", color="gray", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=15)
    ax.set_ylabel("AUC at recovering injected bias")
    ax.set_ylim(0.4, 0.85)
    ax.set_title("How well each detector recovers an injected bias pocket")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMG / "benchmark.png", dpi=130)
    print("wrote", IMG / "benchmark.png")


def bias_map_figure() -> None:
    ds = load_adult()
    rng = np.random.default_rng(3)
    idx = rng.choice(len(ds.y), 3000, replace=False)
    X, dist, y, group = ds.X[idx], ds.distance_X[idx], ds.y[idx], ds.group[idx]
    pocket = ball_pocket(dist, dims=(0, 1, 2, 3), fraction=0.16, seed=3)
    y_bias, _ = inject_label_bias(y, group, pocket, target_group=0, flip_fraction=1.0, seed=3)

    feat = np.column_stack([X, group])
    clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(feat, y_bias)
    scores = clf.predict_proba(feat)[:, 1]
    cf = clf.predict_proba(np.column_stack([X, 1 - group]))[:, 1]
    fair = RandomForestClassifier(n_estimators=200, random_state=0).fit(X, y_bias)
    fair_scores = fair.predict_proba(X)[:, 1]

    ms = fairness_signals_multi(scores, group, dist, y=y, cf_scores=cf, fair_scores=fair_scores)
    score = naive_fusion_score(ms)
    emb = PCA(n_components=2, random_state=0).fit_transform(dist[:, :4])  # the pocket's subspace

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].scatter(emb[~pocket, 0], emb[~pocket, 1], s=8, c="#cbd5e1", label="rest")
    axes[0].scatter(emb[pocket, 0], emb[pocket, 1], s=14, c="#e11d48", label="injected bias")
    axes[0].set_title("Where the bias was injected")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(loc="upper right", fontsize=8)
    sc = axes[1].scatter(emb[:, 0], emb[:, 1], s=11, c=score, cmap="magma")
    axes[1].set_title("The detector's bias score")
    axes[1].set_xlabel("PC1")
    fig.colorbar(sc, ax=axes[1], label="bias score")
    fig.suptitle("Adult: an injected bias pocket and where the simple detector scores it",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(IMG / "bias_map.png", dpi=130, bbox_inches="tight")
    print("wrote", IMG / "bias_map.png")


if __name__ == "__main__":
    benchmark_figure()
    bias_map_figure()
