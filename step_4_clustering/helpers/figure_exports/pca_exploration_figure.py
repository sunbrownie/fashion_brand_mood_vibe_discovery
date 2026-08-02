from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "brand_vibe_mpl"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

try:
    import nbformat as nbf
except Exception:
    nbf = None


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "thesis_text" / "figures"
NOTEBOOK = ROOT / "step_4_clustering" / "pca_latent_axis_exploration.ipynb"


def first_existing(paths: list[Path]) -> Path:
    """Return the first existing path, or the preferred first path if none exist yet."""
    return next((path for path in paths if path.exists()), paths[0])


EMBEDDING_DIR = next(
    (
        path
        for path in [ROOT / "step_2_text_embeddings", ROOT / "text_embeddings"]
        if (path / "brand_embeddings.npz").exists() and (path / "brand_metadata.csv").exists()
    ),
    ROOT / "step_2_text_embeddings",
)
EMBEDDINGS_PATH = EMBEDDING_DIR / "brand_embeddings.npz"
METADATA_PATH = EMBEDDING_DIR / "brand_metadata.csv"
CATEGORIES = ["clothes", "shoes", "bags", "jewellery"]
THRESHOLDS = [0.50, 0.70, 0.80, 0.90, 0.95]
RNG_SEED = 42


def save_current(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=220)
    plt.close()


def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    meta = pd.read_csv(METADATA_PATH)
    embeddings = normalize(np.load(EMBEDDINGS_PATH)["embeddings"].astype("float32"), norm="l2")
    if len(meta) != len(embeddings):
        raise RuntimeError(f"metadata rows ({len(meta)}) do not match embeddings ({len(embeddings)})")
    return meta, embeddings


def components_for_threshold(cumulative: np.ndarray, threshold: float) -> int:
    return int(np.searchsorted(cumulative, threshold) + 1)


def run_pca_summary(meta: pd.DataFrame, embeddings: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    curves = {}
    for category in CATEGORIES:
        X = embeddings[meta["category"].eq(category).to_numpy()]
        pca = PCA(n_components=min(X.shape[0] - 1, X.shape[1]), random_state=RNG_SEED).fit(X)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        curves[category] = cumulative
        row = {"category": category, "n_brands": len(X)}
        for threshold in THRESHOLDS:
            row[f"pcs_for_{int(threshold * 100)}pct"] = components_for_threshold(cumulative, threshold)
        row["pc1_variance"] = float(pca.explained_variance_ratio_[0])
        row["pc2_variance"] = float(pca.explained_variance_ratio_[1])
        row["pc10_cumulative"] = float(cumulative[9])
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "latent_axis_pca_thresholds.csv", index=False)
    return summary, curves


def figure_pca_summary(summary: pd.DataFrame, curves: dict[str, np.ndarray]) -> None:
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
    save_current("latent_axis_pca_summary")


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in cols]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def create_notebook(summary: pd.DataFrame) -> None:
    if nbf is None:
        print("nbformat is not installed; skipping notebook rewrite.")
        return
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# PCA latent-axis exploration\n\n"
            "This notebook checks whether the 768-dimensional brand embedding space has lower-dimensional structure. "
            "PCA is used to measure how many orthogonal directions explain most of the variance."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from sklearn.decomposition import PCA\n"
            "from sklearn.preprocessing import normalize\n\n"
            "ROOT = Path.cwd()\n"
            "if not (ROOT / 'final_dataset').exists() and (ROOT.parent / 'final_dataset').exists():\n"
            "    ROOT = ROOT.parent\n"
            "EMBEDDING_DIR = ROOT / 'step_2_text_embeddings'\n"
            "if not (EMBEDDING_DIR / 'brand_embeddings.npz').exists():\n"
            "    EMBEDDING_DIR = ROOT / 'text_embeddings'\n"
            "embeddings = normalize(np.load(EMBEDDING_DIR / 'brand_embeddings.npz')['embeddings'].astype('float32'))\n"
            "meta = pd.read_csv(EMBEDDING_DIR / 'brand_metadata.csv')\n"
            "meta.shape, embeddings.shape"
        ),
        nbf.v4.new_markdown_cell("## PCA summary\n\n" + markdown_table(summary)),
        nbf.v4.new_code_cell(
            "thresholds = [0.50, 0.70, 0.80, 0.90, 0.95]\n"
            "rows = []\n"
            "for category in ['clothes', 'shoes', 'bags', 'jewellery']:\n"
            "    X = embeddings[meta.category.eq(category).to_numpy()]\n"
            "    pca = PCA(n_components=min(X.shape[0]-1, X.shape[1]), random_state=42).fit(X)\n"
            "    cum = np.cumsum(pca.explained_variance_ratio_)\n"
            "    row = {'category': category, 'n_brands': len(X)}\n"
            "    for t in thresholds:\n"
            "        row[f'pcs_for_{int(t*100)}pct'] = int(np.searchsorted(cum, t) + 1)\n"
            "    rows.append(row)\n"
            "pd.DataFrame(rows)"
        ),
        nbf.v4.new_code_cell(
            "# The exported thesis figure is saved in thesis_text/figures:\n"
            "# - latent_axis_pca_summary.pdf\n"
            "pd.read_csv(ROOT / 'thesis_text' / 'figures' / 'latent_axis_pca_thresholds.csv')"
        ),
    ]
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta, embeddings = load_data()
    summary, curves = run_pca_summary(meta, embeddings)
    figure_pca_summary(summary, curves)
    wrote_notebook = os.environ.get("WRITE_PCA_NOTEBOOK", "1") == "1"
    if wrote_notebook:
        create_notebook(summary)
    print(summary.to_string(index=False))
    if wrote_notebook:
        print(f"Saved notebook to {NOTEBOOK}")
    print(f"Saved figures to {OUT}")


if __name__ == "__main__":
    main()
