from __future__ import annotations

import shutil
import subprocess
import os
import re
import unicodedata
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


def slugify_brand_name(name: str) -> str:
    """Turn a brand name into a safe file-name stem."""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.removeprefix("and_")


def find_moodboard_file(moodboard_root: Path, category: str, name: str, filename: str | float = "") -> Path | None:
    """Find a local moodboard image for a brand, if one exists."""
    if not moodboard_root.exists():
        return None
    if filename and not pd.isna(filename):
        for part in str(filename).split(","):
            part = part.strip()
            if not part:
                continue
            path = moodboard_root / category / part
            if path.exists():
                return path
            matches = list(moodboard_root.glob(f"*/{part}"))
            if matches:
                return matches[0]
    slug = slugify_brand_name(name)
    matches = list((moodboard_root / category).glob(f"*{slug}*.jpg"))
    matches += list((moodboard_root / category).glob(f"*{slug}*.png"))
    matches += list((moodboard_root / category).glob(f"*{slug}*.webp"))
    return matches[0] if matches else None


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repo folder from the repo root or a step notebook folder."""
    start = (start or Path.cwd()).resolve()
    return start.parent if start.name.startswith("step_") else start


def final_dataset_paths(root: Path | None = None) -> dict[str, Path]:
    """List the four clean dataset CSV paths."""
    root = find_repo_root(root)
    return {category: root / "final_dataset" / f"brands_{category}.csv" for category in CATEGORIES}


def missing_final_dataset_files(root: Path | None = None) -> list[Path]:
    """Show which clean dataset CSVs are missing."""
    return [path for path in final_dataset_paths(root).values() if not path.exists()]


def _copy_kaggle_category_csvs(download_dir: Path, root: Path) -> None:
    """Copy the Kaggle CSVs into final_dataset/."""
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
    """Download the clean Kaggle dataset into final_dataset/."""
    root = find_repo_root(root)
    try:
        import kagglehub  # type: ignore

        downloaded = Path(kagglehub.dataset_download(dataset))
    except ImportError:
        target = root / "tmp" / "kaggle_dataset"
        target.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", dataset, "-p", str(target), "--unzip"],
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Install kagglehub or the Kaggle CLI before downloading the dataset. "
                "For example: python -m pip install kagglehub"
            ) from exc
        downloaded = target

    _copy_kaggle_category_csvs(downloaded, root)
    return root / "final_dataset"


def ensure_final_dataset(
    root: Path | None = None,
    *,
    allow_download: bool = False,
    force_download: bool = False,
) -> Path:
    """Make sure final_dataset/ exists, downloading Kaggle if allowed."""
    root = find_repo_root(root)
    if force_download:
        if not allow_download:
            raise ValueError("force_download=True needs allow_download=True.")
        download_kaggle_dataset(root)

    missing = missing_final_dataset_files(root)
    if missing and allow_download:
        download_kaggle_dataset(root)
        missing = missing_final_dataset_files(root)
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Curated category CSVs are missing. Run step 1 or set "
            "DOWNLOAD_DATASET_FROM_KAGGLE=True in the notebook to fetch the Kaggle copy.\n"
            f"{missing_text}"
        )
    return root / "final_dataset"


def load_final_dataset(
    root: Path | None = None,
    *,
    allow_download: bool = False,
    drop_empty_vibe_rows: bool = False,
) -> pd.DataFrame:
    """Load the clean CSVs as one brand table."""
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
    """Count category rows in the frozen analysis snapshot used by clustering."""
    del allow_download
    rows, _, _ = load_embeddings(root)
    counts = rows["category"].value_counts().reindex(CATEGORIES).astype(int)
    return pd.DataFrame(
        [{"category": CATEGORY_LABELS[category], "rows": int(counts.loc[category])} for category in CATEGORIES]
    )


def embedding_dir(root: Path | None = None) -> Path:
    """Find where the saved embeddings live."""
    root = find_repo_root(root)
    return root / "step_2_text_embeddings"


def load_embeddings(root: Path | None = None) -> tuple[pd.DataFrame, np.ndarray, Path]:
    """Load brand metadata and normalised embedding vectors."""
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


def embedding_model_name(root: Path | None = None) -> str:
    """Read the model name saved with the embeddings."""
    emb_dir = embedding_dir(root)
    embeddings_path = emb_dir / "brand_embeddings.npz"
    if not embeddings_path.exists():
        raise FileNotFoundError(
            "Missing embeddings. Run step_2_text_embeddings/brand_text_embeddings.ipynb first."
        )
    archive = np.load(embeddings_path, allow_pickle=True)
    return str(archive["model_name"][0]) if "model_name" in archive.files else "unknown"


def thesis_figures_dir(root: Path | None = None) -> Path:
    """Create and return the figure output folder."""
    root = find_repo_root(root)
    out = root / "thesis_text" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def thesis_tables_dir(root: Path | None = None) -> Path:
    """Create and return the table output folder."""
    out = thesis_figures_dir(root) / THESIS_TABLE_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    return out


def validation_families_table(root: Path | None = None) -> pd.DataFrame:
    """Load the small hand-made validation brand groups."""
    root = find_repo_root(root)
    path = root / "step_4_clustering" / "validation_families_thesis.json"
    table = pd.read_json(path)
    table["brands_included"] = table["brands_included"].map("; ".join)
    table["category"] = table["category"].map(lambda x: CATEGORY_LABELS.get(str(x).lower(), str(x)))
    return table.rename(
        columns={"validation_family": "validation family", "brands_included": "brands included"}
    )


def compute_embedding_validation_summary(root: Path | None = None) -> pd.DataFrame:
    """Check whether known similar brands sit near each other."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.metrics.pairwise import cosine_distances

    root = find_repo_root(root)
    metadata, embeddings, _ = load_embeddings(root)
    families_raw = pd.read_json(root / "step_4_clustering" / "validation_families_thesis.json")
    rows = []

    for category in CATEGORIES:
        category_families = families_raw[families_raw["category"].str.lower().eq(category)]
        category_meta = metadata[metadata["category"].eq(category)].reset_index()
        records = []
        missing = []

        for _, family_row in category_families.iterrows():
            family = str(family_row["validation_family"])
            brands = [str(brand).strip() for brand in family_row["brands_included"] if str(brand).strip()]
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


def pca_threshold_summary(root: Path | None = None) -> pd.DataFrame:
    """Summarise how many PCA dimensions explain the embeddings."""
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
    """Save the PCA summary table."""
    out = thesis_figures_dir(root) / "latent_axis_pca_thresholds.csv"
    pca_threshold_summary(root).to_csv(out, index=False)
    return out


def cluster_selection_table(root: Path | None = None) -> pd.DataFrame:
    """Load the selected cluster counts after step 4 runs."""
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
    """Load the extra clustering-method comparison table."""
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


def save_table(table: pd.DataFrame, name: str, root: Path | None = None) -> Path:
    """Save a notebook table as CSV."""
    path = thesis_tables_dir(root) / f"{name}.csv"
    table.to_csv(path, index=False)
    return path
