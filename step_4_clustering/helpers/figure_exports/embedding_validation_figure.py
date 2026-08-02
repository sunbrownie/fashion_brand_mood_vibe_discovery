from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "thesis_text" / "figures"
SUMMARY_PATH = FIG_DIR / "embedding_validation_summary.csv"

COLORS = {
    "Clothes": "#2E6F95",
    "Shoes": "#D27D2D",
    "Bags": "#4A8F5D",
    "Jewellery": "#9C5FA8",
}


def save_current(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=220)
    plt.close()


def main() -> None:
    df = pd.read_csv(SUMMARY_PATH)
    order = ["Clothes", "Shoes", "Bags", "Jewellery"]
    df["category"] = pd.Categorical(df["category"], categories=order, ordered=True)
    df = df.sort_values("category").reset_index(drop=True)

    x = np.arange(len(df))
    colors = [COLORS[str(cat)] for cat in df["category"]]

    fig, axes = plt.subplots(1, 3, figsize=(12.1, 3.6))

    width = 0.36
    axes[0].bar(x - width / 2, df["same_distance"], width, label="same family", color="#6B8E9D")
    axes[0].bar(x + width / 2, df["different_distance"], width, label="different family", color="#C75C5C")
    axes[0].set_title("Pairwise cosine distance")
    axes[0].set_ylabel("Mean distance")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["category"])
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.18)

    axes[1].bar(x, df["p_at_k_i"], color=colors, alpha=0.86)
    axes[1].set_title(r"Adaptive neighbour precision $P@k_i$")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Mean precision")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["category"])
    axes[1].grid(axis="y", alpha=0.18)
    for idx, value in enumerate(df["p_at_k_i"]):
        axes[1].text(idx, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    metric_width = 0.36
    axes[2].bar(x - metric_width / 2, df["ari"], metric_width, label="ARI", color="#6C5B7B")
    axes[2].bar(x + metric_width / 2, df["nmi"], metric_width, label=r"NMI$_{arith}$", color="#355C7D")
    axes[2].set_title("Cluster agreement")
    axes[2].set_ylim(0, 1.0)
    axes[2].set_ylabel("Score")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(df["category"])
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(axis="y", alpha=0.18)

    fig.suptitle("Embedding validation against manual style families", y=1.04, fontsize=13)
    fig.tight_layout()
    save_current("embedding_validation")


if __name__ == "__main__":
    main()
