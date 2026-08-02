from __future__ import annotations

import shutil
import subprocess
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")


CATEGORIES = ["clothes", "shoes", "bags", "jewellery"]
CATEGORY_LABELS = {
    "clothes": "Clothes",
    "shoes": "Shoes",
    "bags": "Bags",
    "jewellery": "Jewellery",
}
VIBE_COLS = ["aesthetic_keywords", "silhouettes", "materials", "palette"]
KAGGLE_DATASET = "katjazilonova/fashion-brand-mood-vibe-discovery"
THESIS_TABLE_DIRNAME = "thesis_tables"


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root by walking up from a notebook or script folder."""
    start = (start or Path.cwd()).resolve()
    markers = ["final_dataset", "step_1_curate_dataset", "thesis_text"]
    for candidate in [start, *start.parents]:
        if any((candidate / marker).exists() for marker in markers):
            return candidate
    return start


def final_dataset_paths(root: Path | None = None) -> dict[str, Path]:
    """Return the expected curated category CSV paths."""
    root = find_repo_root(root)
    return {category: root / "final_dataset" / f"brands_{category}.csv" for category in CATEGORIES}


def missing_final_dataset_files(root: Path | None = None) -> list[Path]:
    """Return curated category CSVs that are not present locally."""
    return [path for path in final_dataset_paths(root).values() if not path.exists()]


def _copy_kaggle_category_csvs(download_dir: Path, root: Path) -> None:
    final_dir = root / "final_dataset"
    final_dir.mkdir(parents=True, exist_ok=True)
    for category in CATEGORIES:
        matches = sorted(download_dir.rglob(f"brands_{category}.csv"))
        if not matches:
            raise FileNotFoundError(
                f"Kaggle download did not contain brands_{category}.csv under {download_dir}"
            )
        shutil.copy2(matches[0], final_dir / f"brands_{category}.csv")


def download_kaggle_dataset(root: Path | None = None, dataset: str = KAGGLE_DATASET) -> Path:
    """Download the curated Kaggle dataset and populate final_dataset/."""
    root = find_repo_root(root)
    try:
        import kagglehub  # type: ignore

        downloaded = Path(kagglehub.dataset_download(dataset))
    except Exception:
        target = root / "tmp" / "kaggle_dataset"
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", str(target), "--unzip"],
            check=True,
        )
        downloaded = target

    _copy_kaggle_category_csvs(downloaded, root)
    return root / "final_dataset"


def ensure_final_dataset(root: Path | None = None, *, allow_download: bool = False) -> Path:
    """Return final_dataset/, optionally downloading it from Kaggle if missing."""
    root = find_repo_root(root)
    missing = missing_final_dataset_files(root)
    if missing and allow_download:
        download_kaggle_dataset(root)
        missing = missing_final_dataset_files(root)
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Curated category CSVs are missing. Run step 1 or set "
            "ALLOW_KAGGLE_DOWNLOAD=True in the notebook to fetch the Kaggle copy.\n"
            f"{missing_text}"
        )
    return root / "final_dataset"


def load_final_dataset(
    root: Path | None = None,
    *,
    allow_download: bool = False,
    drop_empty_vibe_rows: bool = False,
) -> pd.DataFrame:
    """Load the curated category CSVs as one analysis table."""
    root = find_repo_root(root)
    ensure_final_dataset(root, allow_download=allow_download)
    frames = []
    for category, path in final_dataset_paths(root).items():
        df = pd.read_csv(path)
        df["category"] = category
        frames.append(df)
    rows = pd.concat(frames, ignore_index=True)
    if drop_empty_vibe_rows:
        rows = rows.dropna(subset=[c for c in VIBE_COLS if c in rows.columns], how="all")
    return rows.reset_index(drop=True)


def dataset_category_counts(root: Path | None = None, *, allow_download: bool = False) -> pd.DataFrame:
    """Return the thesis dataset-count table."""
    rows = load_final_dataset(root, allow_download=allow_download)
    counts = rows["category"].value_counts().reindex(CATEGORIES).astype(int)
    return pd.DataFrame(
        [{"category": CATEGORY_LABELS[category], "rows": int(counts.loc[category])} for category in CATEGORIES]
    )


def embedding_dir(root: Path | None = None) -> Path:
    """Return the preferred embedding folder, using the numbered step if present."""
    root = find_repo_root(root)
    candidates = [root / "step_2_text_embeddings", root / "text_embeddings"]
    for candidate in candidates:
        if (candidate / "brand_embeddings.npz").exists() and (candidate / "brand_metadata.csv").exists():
            return candidate
    return candidates[0]


def load_embeddings(root: Path | None = None) -> tuple[pd.DataFrame, np.ndarray, Path]:
    """Load embedding metadata and L2-normalised vectors."""
    from sklearn.preprocessing import normalize

    emb_dir = embedding_dir(root)
    embeddings_path = emb_dir / "brand_embeddings.npz"
    metadata_path = emb_dir / "brand_metadata.csv"
    if not embeddings_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Missing embeddings. Run step_2_text_embeddings/brand_text_embeddings.ipynb first."
        )
    archive = np.load(embeddings_path, allow_pickle=True)
    embeddings = normalize(archive["embeddings"].astype("float32"), norm="l2")
    metadata = pd.read_csv(metadata_path).reset_index(drop=True)
    if len(metadata) != len(embeddings):
        raise RuntimeError(
            f"Embedding metadata rows ({len(metadata)}) do not match vectors ({len(embeddings)})."
        )
    return metadata, embeddings, emb_dir


def thesis_figures_dir(root: Path | None = None) -> Path:
    """Return thesis_text/figures, creating it if needed."""
    root = find_repo_root(root)
    out = root / "thesis_text" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def thesis_tables_dir(root: Path | None = None) -> Path:
    """Return a folder for notebook-exported thesis tables."""
    out = thesis_figures_dir(root) / THESIS_TABLE_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    return out


def validation_families_table(root: Path | None = None) -> pd.DataFrame:
    """Return the manual validation-family table used in the thesis."""
    root = find_repo_root(root)
    path = root / "step_4_clustering" / "validation_families_thesis.csv"
    table = pd.read_csv(path)
    table["category"] = table["category"].map(lambda x: CATEGORY_LABELS.get(str(x).lower(), str(x)))
    return table.rename(
        columns={"validation_family": "validation family", "brands_included": "brands included"}
    )


def compute_embedding_validation_summary(root: Path | None = None) -> pd.DataFrame:
    """Compute the thesis embedding-validation table from validation families and embeddings."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.metrics.pairwise import cosine_distances

    root = find_repo_root(root)
    metadata, embeddings, _ = load_embeddings(root)
    families_raw = pd.read_csv(root / "step_4_clustering" / "validation_families_thesis.csv")
    rows = []

    for category in CATEGORIES:
        category_families = families_raw[families_raw["category"].str.lower().eq(category)]
        category_meta = metadata[metadata["category"].eq(category)].reset_index()
        records = []
        missing = []

        for _, family_row in category_families.iterrows():
            family = str(family_row["validation_family"])
            brands = [brand.strip() for brand in str(family_row["brands_included"]).split(";") if brand.strip()]
            for brand in brands:
                match = category_meta[category_meta["brand_name"].str.lower().eq(brand.lower())]
                if match.empty:
                    missing.append(brand)
                    continue
                source_idx = int(match.iloc[0]["index"])
                records.append(
                    {
                        "brand_name": match.iloc[0]["brand_name"],
                        "expected_family": family,
                        "source_idx": source_idx,
                    }
                )

        validation = pd.DataFrame(records)
        if validation.empty:
            continue
        X = embeddings[validation["source_idx"].to_numpy()]
        distances = cosine_distances(X)
        same_distances = []
        different_distances = []
        for i, j in combinations(range(len(validation)), 2):
            if validation.loc[i, "expected_family"] == validation.loc[j, "expected_family"]:
                same_distances.append(distances[i, j])
            else:
                different_distances.append(distances[i, j])

        family_sizes = validation["expected_family"].value_counts().to_dict()
        precisions = []
        for i, row in validation.iterrows():
            k = min(3, int(family_sizes[row["expected_family"]]) - 1)
            if k <= 0:
                continue
            order = np.argsort(distances[i])
            neighbours = [idx for idx in order if idx != i][:k]
            precision = np.mean(validation.loc[neighbours, "expected_family"].eq(row["expected_family"]))
            precisions.append(float(precision))

        labels_true = pd.Categorical(validation["expected_family"]).codes
        n_families = int(validation["expected_family"].nunique())
        labels_pred = KMeans(n_clusters=n_families, n_init=50, random_state=42).fit_predict(X)

        rows.append(
            {
                "category": CATEGORY_LABELS[category],
                "brands": int(len(validation)),
                "families": n_families,
                "same_distance": float(np.mean(same_distances)),
                "different_distance": float(np.mean(different_distances)),
                "p_at_k_i": float(np.mean(precisions)),
                "ari": float(adjusted_rand_score(labels_true, labels_pred)),
                "nmi": float(
                    normalized_mutual_info_score(labels_true, labels_pred, average_method="arithmetic")
                ),
                "missing_validation_brands": "; ".join(sorted(set(missing))),
            }
        )

    return pd.DataFrame(rows)


def save_embedding_validation_summary(root: Path | None = None) -> Path:
    """Compute and save embedding_validation_summary.csv for the thesis figure script."""
    out = thesis_figures_dir(root) / "embedding_validation_summary.csv"
    compute_embedding_validation_summary(root).to_csv(out, index=False)
    return out


def pca_threshold_summary(root: Path | None = None) -> pd.DataFrame:
    """Compute the thesis PCA threshold table from saved embeddings."""
    from sklearn.decomposition import PCA

    metadata, embeddings, _ = load_embeddings(root)
    thresholds = [0.50, 0.70, 0.80, 0.90, 0.95]
    rows = []
    for category in CATEGORIES:
        X = embeddings[metadata["category"].eq(category).to_numpy()]
        pca = PCA(n_components=min(X.shape[0] - 1, X.shape[1]), random_state=42).fit(X)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        row = {"category": CATEGORY_LABELS[category], "n_brands": len(X)}
        for threshold in thresholds:
            row[f"pcs_for_{int(threshold * 100)}pct"] = int(np.searchsorted(cumulative, threshold) + 1)
        row["pc1_variance"] = float(pca.explained_variance_ratio_[0])
        row["pc2_variance"] = float(pca.explained_variance_ratio_[1])
        row["pc10_cumulative"] = float(cumulative[9])
        rows.append(row)
    return pd.DataFrame(rows)


def save_pca_threshold_summary(root: Path | None = None) -> Path:
    """Compute and save latent_axis_pca_thresholds.csv for the thesis."""
    out = thesis_figures_dir(root) / "latent_axis_pca_thresholds.csv"
    pca_threshold_summary(root).to_csv(out, index=False)
    return out


def cluster_selection_table(root: Path | None = None) -> pd.DataFrame:
    """Return the thesis cluster-selection table from exported clustering CSVs."""
    root = find_repo_root(root)
    out = thesis_figures_dir(root)
    selected_path = out / "cluster_k_selected.csv"
    if not selected_path.exists():
        raise FileNotFoundError(
            "Missing cluster_k_selected.csv; expected after the clustering figure helper "
            "has created the saved summaries."
        )
    selected = pd.read_csv(selected_path)
    counts = dataset_category_counts(root).set_index("category")["rows"].to_dict()
    selected["category_label"] = selected["category"].map(CATEGORY_LABELS)
    selected["brands"] = selected["category_label"].map(counts).astype(int)
    return selected[
        [
            "category_label",
            "brands",
            "selected_k",
            "silhouette_best_k",
            "davies_bouldin_best_k",
            "selected_silhouette",
            "selected_davies_bouldin",
            "selected_score",
        ]
    ].rename(
        columns={
            "category_label": "category",
            "selected_k": "selected k",
            "silhouette_best_k": "silhouette-best k",
            "davies_bouldin_best_k": "DB-best k",
            "selected_silhouette": "selected silhouette",
            "selected_davies_bouldin": "selected DB",
            "selected_score": "combined score",
        }
    )


def cluster_algorithm_table(root: Path | None = None) -> pd.DataFrame:
    """Return the supplementary clustering-algorithm comparison table."""
    path = thesis_figures_dir(root) / "cluster_algorithm_scores.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Missing cluster_algorithm_scores.csv. Run the clustering notebook export cell first."
        )
    table = pd.read_csv(path)
    table["category"] = table["category"].map(CATEGORY_LABELS)
    return table.rename(
        columns={
            "method": "method",
            "n_clusters_ex_noise": "clusters",
            "noise_points": "noise",
            "silhouette_cosine": "cosine silhouette",
        }
    )


def recommender_feedback_table() -> pd.DataFrame:
    """Return the small informal recommender-feedback table from the thesis."""
    return pd.DataFrame(
        [
            {
                "user": "User 1",
                "first useful round": "Round 3",
                "qualitative feedback": "The recommender started showing styles and brands the user preferred.",
            },
            {
                "user": "User 2",
                "first useful round": "Round 3",
                "qualitative feedback": (
                    "The recommender was judged very relevant for discovery, although the brand set should be "
                    "expanded, male clothing coverage felt restricted, and some male moodboards still showed "
                    "female clothing."
                ),
            },
            {
                "user": "User 3",
                "first useful round": "Round 5",
                "qualitative feedback": (
                    "The recommender was also judged very relevant for discovering brands through style rather "
                    "than brand-name familiarity, and the moodboards were found useful."
                ),
            },
        ]
    )


def save_table(table: pd.DataFrame, name: str, root: Path | None = None) -> Path:
    """Save a notebook-computed thesis table as CSV."""
    path = thesis_tables_dir(root) / f"{name}.csv"
    table.to_csv(path, index=False)
    return path
