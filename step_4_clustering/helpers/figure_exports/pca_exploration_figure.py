from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[3]
FIGURE_HELPERS_DIR = Path(__file__).resolve().parent
if str(FIGURE_HELPERS_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURE_HELPERS_DIR))
SOURCE_HELPERS_DIR = ROOT / "helpers"
if str(SOURCE_HELPERS_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_HELPERS_DIR))

from figure_export_helpers import save_current_figure
from source_helpers import CATEGORIES, load_embeddings, pca_threshold_summary, save_pca_threshold_summary

OUT = ROOT / "thesis_text" / "figures"
RNG_SEED = 42


def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load metadata and embedding vectors."""
    meta, embeddings, _ = load_embeddings(ROOT)
    return meta, embeddings


def components_for_threshold(cumulative: np.ndarray, threshold: float) -> int:
    """Find how many components reach a variance threshold."""
    return int(np.searchsorted(cumulative, threshold) + 1)


def run_pca_summary(meta: pd.DataFrame, embeddings: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Run PCA curves and use the shared helper for the summary table."""
    curves = {}
    for category in CATEGORIES:
        X = embeddings[meta["category"].eq(category).to_numpy()]
        pca = PCA(n_components=min(X.shape[0] - 1, X.shape[1]), random_state=RNG_SEED).fit(X)
        curves[category] = np.cumsum(pca.explained_variance_ratio_)
    save_pca_threshold_summary(ROOT)
    return pca_threshold_summary(ROOT), curves


def figure_pca_summary(summary: pd.DataFrame, curves: dict[str, np.ndarray]) -> None:
    """Draw the PCA summary figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    clothes_curve = curves["clothes"]
    axes[0].plot(np.arange(1, len(clothes_curve) + 1), clothes_curve, color="#2E6F95", linewidth=2.2)
    for threshold in [0.50, 0.80, 0.90, 0.95]:
        k = components_for_threshold(clothes_curve, threshold)
        axes[0].axhline(threshold, color="#9AA6B2", linestyle="--", linewidth=0.8)
        axes[0].axvline(k, color="#9AA6B2", linestyle=":", linewidth=0.8)
        axes[0].text(k + 2, threshold + 0.012, f"{int(threshold * 100)}%: {k} PCs", fontsize=8)
    axes[0].set_xlim(1, 125)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Clothes PCA cumulative explained variance")
    axes[0].set_xlabel("number of principal components")
    axes[0].set_ylabel("cumulative explained variance")
    axes[0].grid(alpha=0.18)

    x = np.arange(len(summary))
    width = 0.24
    axes[1].bar(x - width, summary["pcs_for_50pct"], width, label="50%", color="#6B8E9D")
    axes[1].bar(x, summary["pcs_for_80pct"], width, label="80%", color="#D6A04D")
    axes[1].bar(x + width, summary["pcs_for_90pct"], width, label="90%", color="#8B6FA9")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary["category"])
    axes[1].set_title("Components needed by category")
    axes[1].set_ylabel("principal components")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.18)
    for i, row in enumerate(summary.itertuples()):
        axes[1].text(i - width, row.pcs_for_50pct + 1.5, str(row.pcs_for_50pct), ha="center", fontsize=8)
        axes[1].text(i, row.pcs_for_80pct + 1.5, str(row.pcs_for_80pct), ha="center", fontsize=8)
        axes[1].text(i + width, row.pcs_for_90pct + 1.5, str(row.pcs_for_90pct), ha="center", fontsize=8)

    plt.tight_layout()
    save_current_figure(OUT, "latent_axis_pca_summary")


def main() -> None:
    """Rebuild the PCA table and figure."""
    OUT.mkdir(parents=True, exist_ok=True)
    meta, embeddings = load_data()
    summary, curves = run_pca_summary(meta, embeddings)
    figure_pca_summary(summary, curves)
    print(summary.to_string(index=False))
    print(f"Saved figures to {OUT}")


if __name__ == "__main__":
    main()
