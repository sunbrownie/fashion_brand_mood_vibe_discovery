from __future__ import annotations

import os
import sys
import textwrap
import argparse
from collections import Counter
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

try:
    import umap
except Exception:
    umap = None


ROOT = Path(__file__).resolve().parents[3]
FIGURE_HELPERS_DIR = Path(__file__).resolve().parent
if str(FIGURE_HELPERS_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURE_HELPERS_DIR))
SHARED_HELPERS_DIR = ROOT / "step_4_clustering" / "helpers"
if str(SHARED_HELPERS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_HELPERS_DIR))
SOURCE_HELPERS_DIR = ROOT / "helpers"
if str(SOURCE_HELPERS_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_HELPERS_DIR))

from figure_export_helpers import save_current_figure
from source_helpers import CATEGORIES, final_dataset_paths, find_moodboard_file, load_embeddings
from shared_cluster_helpers import (
    CLUSTER_TITLES,
    build_category_dataset,
    minmax,
    representative_brands,
    separate_text_labels,
    split_terms,
    tokenize_style_text,
)

OUT = ROOT / "thesis_text" / "figures"
MOODBOARDS = ROOT / "moodboards_creation" / "moodboards"

CSV_FILES = final_dataset_paths(ROOT)
VIBE_COLS = ["aesthetic_keywords", "silhouettes", "materials", "palette"]
MOODBOARD_COLUMNS = ["category", "cluster", "brand_name", "moodboard_path"]
K_RANGES = {
    "clothes": range(7, 23),
    "shoes": range(7, 26),
    "bags": range(7, 23),
    "jewellery": range(7, 23),
}
K_WEIGHTS = {"silhouette": 0.45, "davies_bouldin": 0.35, "granularity": 0.20}
RNG_SEED = 42

COLORS = {
    "clothes": "#2E6F95",
    "shoes": "#D27D2D",
    "bags": "#4A8F5D",
    "jewellery": "#9C5FA8",
}

def load_moodboard_lookup() -> dict[tuple[str, str], str]:
    """Read saved moodboard filenames from the clean CSVs."""
    lookup: dict[tuple[str, str], str] = {}
    for category, path in CSV_FILES.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "moodboard" not in df.columns:
            continue
        for _, row in df.iterrows():
            key = (category, str(row.get("brand_name", "")).strip().lower())
            moodboard = row.get("moodboard", "")
            if key[1] and not pd.isna(moodboard) and str(moodboard).strip():
                lookup[key] = str(moodboard).strip()
    return lookup


def load_source() -> tuple[pd.DataFrame, np.ndarray]:
    """Load embedding rows and attach any local moodboard names."""
    rows, embeddings, _ = load_embeddings(ROOT)
    rows = rows.reset_index(drop=True)
    moodboard_lookup = load_moodboard_lookup()
    rows["moodboard"] = [
        moodboard_lookup.get((str(row.category), str(row.brand_name).strip().lower()), "")
        for row in rows.itertuples()
    ]
    rows["row_id"] = np.arange(len(rows))
    if len(rows) != len(embeddings):
        raise RuntimeError(f"Rows ({len(rows)}) do not match embeddings ({len(embeddings)}).")
    return rows, embeddings


def choose_k(brands: pd.DataFrame, X: np.ndarray, category: str) -> tuple[int, pd.DataFrame, dict[str, np.ndarray]]:
    """Try possible k values and choose the best cluster count."""
    rows = []
    labels_by_k = {}
    for k in K_RANGES[category]:
        labels = KMeans(n_clusters=k, n_init=30, random_state=RNG_SEED).fit_predict(X)
        labels_by_k[k] = labels
        rows.append({
            "category": category,
            "k": k,
            "silhouette_cosine": silhouette_score(X, labels, metric="cosine"),
            "davies_bouldin": davies_bouldin_score(X, labels),
        })
    scores = pd.DataFrame(rows)
    scores["silhouette_norm"] = minmax(scores["silhouette_cosine"], True)
    scores["davies_bouldin_norm"] = minmax(scores["davies_bouldin"], False)
    scores["granularity_norm"] = minmax(scores["k"], True)
    scores["selection_score"] = (
        K_WEIGHTS["silhouette"] * scores["silhouette_norm"]
        + K_WEIGHTS["davies_bouldin"] * scores["davies_bouldin_norm"]
        + K_WEIGHTS["granularity"] * scores["granularity_norm"]
    )
    best_score = scores["selection_score"].max()
    near_best = scores[scores["selection_score"] >= best_score - 0.03]
    selected = int(near_best.sort_values(["k", "selection_score"], ascending=[False, False]).iloc[0]["k"])
    return selected, scores, labels_by_k


def phrase_counter(rows: pd.DataFrame, column: str) -> Counter:
    """Count short repeated phrases in one brand-card column."""
    counter = Counter()
    for _, row in rows.iterrows():
        for term in split_terms(row[column]):
            words = tokenize_style_text(term)
            if words:
                counter[" ".join(words[:3])] += 1
    return counter


def build_cluster_summary(category: str, brands: pd.DataFrame, X: np.ndarray, labels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create cluster titles, terms, examples, and moodboard rows."""
    reps = representative_brands(brands, X, labels, top_n=8)
    centroids = {}
    for cluster in sorted(set(labels)):
        idx = np.where(labels == cluster)[0]
        centroids[int(cluster)] = normalize(X[idx].mean(axis=0, keepdims=True))[0]
    centroid_ids = list(centroids)
    centroid_matrix = np.vstack([centroids[c] for c in centroid_ids])
    centroid_sims = cosine_similarity(centroid_matrix)
    np.fill_diagonal(centroid_sims, -np.inf)

    summary_rows = []
    mood_rows = []
    for cluster in sorted(set(labels)):
        cluster = int(cluster)
        sub = brands.loc[labels == cluster].copy()
        title = CLUSTER_TITLES.get((category, cluster), f"Cluster {cluster}")
        aesthetic = phrase_counter(sub, "aesthetic_keywords").most_common(8)
        silhouette = phrase_counter(sub, "silhouettes").most_common(5)
        materials = phrase_counter(sub, "materials").most_common(4)
        palette = phrase_counter(sub, "palette").most_common(4)
        top_terms = [term for term, _ in aesthetic[:5]]
        support_terms = [term for term, _ in silhouette[:3] + materials[:2] + palette[:2]]
        examples = reps[reps["cluster"].eq(cluster)]["brand_name"].head(6).tolist()

        row_i = centroid_ids.index(cluster)
        nearest_order = np.argsort(-centroid_sims[row_i])[:2]
        nearest = [(centroid_ids[i], float(centroid_sims[row_i, i])) for i in nearest_order if np.isfinite(centroid_sims[row_i, i])]

        summary_rows.append({
            "category": category,
            "cluster": cluster,
            "n_brands": int(len(sub)),
            "cluster_title": title,
            "top_aesthetic_terms": ", ".join(top_terms),
            "supporting_terms": ", ".join(support_terms),
            "example_brands": ", ".join(examples),
            "nearest_cluster": nearest[0][0] if nearest else "",
            "nearest_similarity": nearest[0][1] if nearest else np.nan,
            "second_nearest_cluster": nearest[1][0] if len(nearest) > 1 else "",
            "second_nearest_similarity": nearest[1][1] if len(nearest) > 1 else np.nan,
        })

        brand_moodboards = brands.set_index("brand_name")["moodboard"].fillna("").to_dict()
        for name in examples:
            path = find_moodboard_file(MOODBOARDS, category, name, brand_moodboards.get(name, ""))
            if path is not None:
                mood_rows.append({"category": category, "cluster": cluster, "brand_name": name, "moodboard_path": str(path)})
    return pd.DataFrame(summary_rows), pd.DataFrame(mood_rows, columns=MOODBOARD_COLUMNS)


def figure_k_selection(scores: pd.DataFrame, selected: pd.DataFrame) -> None:
    """Draw the figure that shows why each k was selected."""
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3), sharex=False, sharey=True)
    for ax, category in zip(axes.ravel(), CATEGORIES):
        sub = scores[scores["category"].eq(category)]
        plot = sub.melt(
            id_vars=["k"],
            value_vars=["silhouette_norm", "davies_bouldin_norm", "selection_score"],
            var_name="criterion",
            value_name="normalized_score",
        )
        labels = {
            "silhouette_norm": "silhouette",
            "davies_bouldin_norm": "Davies-Bouldin",
            "selection_score": "combined",
        }
        plot["criterion"] = plot["criterion"].map(labels)
        sns.lineplot(data=plot, x="k", y="normalized_score", hue="criterion", marker="o", ax=ax)
        k = int(selected[selected["category"].eq(category)]["selected_k"].iloc[0])
        ax.axvline(k, color="#E76F51", linestyle=":", linewidth=2.2)
        ax.set_title(f"{category.title()}: selected k={k}")
        ax.set_xlabel("number of clusters")
        ax.set_ylabel("normalized score")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    save_current_figure(OUT, "cluster_selection_scores", formats=("pdf", "png", "jpg"))


def figure_cluster_maps(results: dict[str, dict[str, object]]) -> None:
    """Draw the main cluster map for each category."""
    if umap is None:
        raise ImportError("UMAP is required to regenerate Figure 4.")
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.0))
    for ax, category in zip(axes.ravel(), CATEGORIES):
        brands = results[category]["brands"]
        X = results[category]["X"]
        labels = results[category]["labels"]
        summary = results[category]["summary"]
        reducer = umap.UMAP(
            n_neighbors=min(35, max(2, len(X) - 1)),
            min_dist=0.05,
            metric="cosine",
            random_state=RNG_SEED,
            spread=1.2,
        )
        coords = reducer.fit_transform(X)
        projection_name = "UMAP"
        cmap = plt.get_cmap("tab20")
        title_lookup = {
            int(row.cluster): str(row.cluster_title)
            for row in summary.itertuples()
        }
        texts = []
        for cluster in sorted(set(labels)):
            mask = labels == cluster
            ax.scatter(coords[mask, 0], coords[mask, 1], s=9, alpha=0.60, color=cmap(int(cluster) % 20))
            center = np.median(coords[mask], axis=0)
            cluster_name = title_lookup.get(int(cluster), f"Cluster {int(cluster)}")
            label = textwrap.fill(f"{int(cluster)} {cluster_name}", width=16)
            texts.append(
                ax.text(
                    center[0],
                    center[1],
                    label,
                    fontsize=6.9,
                    ha="center",
                    va="center",
                    linespacing=0.88,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="0.45", alpha=0.90),
                    zorder=5,
                )
            )
        ax.margins(0.14)
        separate_text_labels(ax, texts, pad=3.2, max_iter=260)
        ax.set_title(f"{category.title()} clusters (k={len(set(labels))})", fontsize=10.0)
        ax.set_xlabel(f"{projection_name}-1", fontsize=8.8)
        ax.set_ylabel(f"{projection_name}-2", fontsize=8.8)
        ax.tick_params(axis="both", labelsize=8.0)
        ax.grid(alpha=0.14)
    plt.tight_layout(pad=1.0, w_pad=1.1, h_pad=1.2)
    save_current_figure(OUT, "cluster_maps_by_category", formats=("pdf", "png", "jpg"))


def figure_size_summary(summary_all: pd.DataFrame) -> None:
    """Draw how big each cluster is."""
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.5))
    for ax, category in zip(axes.ravel(), CATEGORIES):
        sub = summary_all[summary_all["category"].eq(category)].sort_values("n_brands")
        labels = [f"{r.cluster}: {r.cluster_title}" for r in sub.itertuples()]
        ax.barh(labels, sub["n_brands"], color=COLORS[category], alpha=0.82)
        ax.set_title(f"{category.title()} cluster sizes")
        ax.set_xlabel("brands")
        ax.tick_params(axis="y", labelsize=6.8)
        ax.grid(axis="x", alpha=0.16)
    plt.tight_layout()
    save_current_figure(OUT, "cluster_size_summary", formats=("pdf", "png", "jpg"))


def moodboard_thumbnail(path: str) -> Image.Image:
    """Make a small moodboard image for cluster cards."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((420, 760), Image.Resampling.LANCZOS)
    return img


def draw_cluster_card(ax, category: str, row: pd.Series, moodboards: pd.DataFrame) -> None:
    """Draw one cluster card with terms, examples, and images."""
    ax.set_facecolor("#FAFAF7")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CFC8BC")
        spine.set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    title = textwrap.fill(f"{int(row.cluster)}. {row.cluster_title} ({int(row.n_brands)})", width=34)
    ax.text(0.03, 0.96, title, ha="left", va="top", fontsize=9.2, weight="bold", color="#1f2933")
    ax.text(0.03, 0.82, "Representative brands", ha="left", va="top", fontsize=6.5, weight="bold", color="#5d6670")
    examples = textwrap.fill(str(row.example_brands), width=29)
    ax.text(0.03, 0.77, examples, ha="left", va="top", fontsize=6.1, color="#2f3b45")

    terms = [t.strip() for t in str(row.top_aesthetic_terms).split(",") if t.strip()][:5]
    support = [t.strip() for t in str(row.supporting_terms).split(",") if t.strip()][:4]
    all_terms = terms + support
    for i, term in enumerate(all_terms[:7]):
        y = 0.46 - i * 0.043
        width = 0.17 + 0.011 * max(0, 7 - i)
        ax.add_patch(plt.Rectangle((0.03, y - 0.025), width, 0.030, color=COLORS[category], alpha=0.20 + 0.060 * max(0, 6 - i)))
        ax.text(0.04, y - 0.010, textwrap.shorten(term, width=26, placeholder="..."), fontsize=6.1, ha="left", va="center", color="#1f2933")

    sub = moodboards[(moodboards["category"].eq(category)) & (moodboards["cluster"].eq(int(row.cluster)))].head(3)
    image_slots = [(0.32, 0.54), (0.55, 0.77), (0.78, 0.985)]
    for (x0, x1), (_, mood) in zip(image_slots, sub.iterrows()):
        try:
            img = moodboard_thumbnail(mood["moodboard_path"])
        except Exception:
            continue
        ax.imshow(np.asarray(img), extent=(x0, x1, 0.13, 0.79), aspect="auto", interpolation="lanczos", zorder=1)
        ax.text((x0 + x1) / 2, 0.105, textwrap.fill(str(mood["brand_name"]), width=12), ha="center", va="top", fontsize=5.4)

    nearest_title = getattr(row, "nearest_cluster_title", "")
    second_title = getattr(row, "second_nearest_cluster_title", "")
    near = f"closest: {int(row.nearest_cluster)}. {nearest_title}"
    second = f"2nd closest: {int(row.second_nearest_cluster)}. {second_title}"
    ax.text(0.03, 0.075, textwrap.fill(near, width=30), ha="left", va="bottom", fontsize=5.6, color="#5d6670")
    ax.text(0.03, 0.030, textwrap.fill(second, width=30), ha="left", va="bottom", fontsize=5.6, color="#5d6670")


def add_nearest_cluster_titles(summary: pd.DataFrame) -> pd.DataFrame:
    """Add readable titles for the closest neighbouring clusters."""
    summary = summary.copy()
    lookup = {
        (str(row.category), int(row.cluster)): str(row.cluster_title)
        for row in summary.itertuples()
    }
    summary["nearest_cluster_title"] = [
        lookup.get((str(row.category), int(row.nearest_cluster)), "")
        for row in summary.itertuples()
    ]
    summary["second_nearest_cluster_title"] = [
        lookup.get((str(row.category), int(row.second_nearest_cluster)), "")
        for row in summary.itertuples()
    ]
    return summary


def figure_cluster_cards(category: str, summary: pd.DataFrame, moodboards: pd.DataFrame) -> None:
    """Save all cluster-card pages for one category."""
    summary = add_nearest_cluster_titles(summary)
    rows = (
        summary[summary["category"].eq(category) & summary["n_brands"].gt(1)]
        .sort_values("cluster")
        .reset_index(drop=True)
    )
    cards_per_page = 6
    for part, start in enumerate(range(0, len(rows), cards_per_page), start=1):
        page = rows.iloc[start:start + cards_per_page]
        fig, axes = plt.subplots(3, 2, figsize=(12.6, 14.8))
        for ax, (_, row) in zip(axes.ravel(), page.iterrows()):
            draw_cluster_card(ax, category, row, moodboards)
        for ax in axes.ravel()[len(page):]:
            ax.axis("off")
        fig.suptitle(f"{category.title()} cluster cards, part {part}", fontsize=16, weight="bold", y=0.995)
        plt.tight_layout(rect=(0, 0, 1, 0.985))
        save_current_figure(OUT, f"cluster_cards_{category}_part{part}", formats=("pdf", "png", "jpg"))


def figure_main_cluster_card_examples(summary: pd.DataFrame, moodboards: pd.DataFrame) -> None:
    """Save a small figure with a few example cluster cards."""
    summary = add_nearest_cluster_titles(summary)
    examples = [
        ("clothes", 5),
        ("clothes", 13),
        ("jewellery", 6),
        ("shoes", 0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 10.8))
    for ax, (category, cluster) in zip(axes.ravel(), examples):
        row = summary[(summary["category"].eq(category)) & (summary["cluster"].eq(cluster))].iloc[0]
        draw_cluster_card(ax, category, row, moodboards)
        ax.set_title(category.title(), fontsize=12, weight="bold", pad=10)
    fig.suptitle("Example cluster cards from category-specific embedding spaces", fontsize=14, weight="bold", y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.975))
    save_current_figure(OUT, "cluster_card_examples_main", formats=("pdf", "png", "jpg"))


def algorithm_scores(category: str, X: np.ndarray, selected_k: int, labels: np.ndarray) -> list[dict[str, object]]:
    """Compare k-means with a couple of clustering alternatives."""
    rows = []
    methods = {
        f"k-means k={selected_k}": labels,
        f"agglomerative k={selected_k}": AgglomerativeClustering(n_clusters=selected_k, metric="cosine", linkage="average").fit_predict(X),
    }
    try:
        dims = min(50, X.shape[1] - 1, len(X) - 1)
        X_svd = normalize(TruncatedSVD(n_components=dims, random_state=RNG_SEED).fit_transform(X))
        hdb = HDBSCAN(min_cluster_size=max(6, len(X) // 70), min_samples=max(3, len(X) // 180), metric="cosine").fit_predict(X_svd)
        n_hdb = len(set(hdb) - {-1})
        if 2 <= n_hdb <= max(K_RANGES[category]):
            methods["HDBSCAN"] = hdb
            hdb_data = X_svd
        else:
            hdb_data = None
    except Exception:
        hdb_data = None

    for method, method_labels in methods.items():
        method_labels = np.asarray(method_labels)
        mask = method_labels != -1
        data = hdb_data if method == "HDBSCAN" and hdb_data is not None else X
        score = np.nan
        if len(set(method_labels[mask])) > 1 and mask.sum() > 2:
            score = silhouette_score(data[mask], method_labels[mask], metric="cosine")
        rows.append({
            "category": category,
            "method": method,
            "n_clusters_ex_noise": len(set(method_labels) - {-1}),
            "noise_points": int(np.sum(method_labels == -1)),
            "silhouette_cosine": score,
        })
    return rows


def main(main_figures_only: bool = False, cards_category: str | None = None) -> None:
    """Run the clustering export and save the figure files."""
    OUT.mkdir(parents=True, exist_ok=True)
    if cards_category is not None:
        for old_card in OUT.glob(f"cluster_cards_{cards_category}_part*.*"):
            old_card.unlink()
    elif not main_figures_only:
        for old_card in OUT.glob("cluster_cards_*_part*.*"):
            old_card.unlink()
    source_rows, embeddings = load_source()
    all_scores = []
    selected_rows = []
    summary_rows = []
    mood_rows = []
    algo_rows = []
    results: dict[str, dict[str, object]] = {}
    categories_to_run = [cards_category] if cards_category is not None and main_figures_only else CATEGORIES

    for category in categories_to_run:
        print(f"Running clustering export for {category}...", flush=True)
        brands, X = build_category_dataset(source_rows, embeddings, category)
        selected_k, scores, labels_by_k = choose_k(brands, X, category)
        labels = labels_by_k[selected_k]
        summary, moodboards = build_cluster_summary(category, brands, X, labels)
        summary["selected_k"] = selected_k
        results[category] = {"brands": brands, "X": X, "labels": labels, "summary": summary}

        all_scores.append(scores)
        selected_rows.append({
            "category": category,
            "selected_k": selected_k,
            "silhouette_best_k": int(scores.sort_values("silhouette_cosine", ascending=False).iloc[0]["k"]),
            "davies_bouldin_best_k": int(scores.sort_values("davies_bouldin").iloc[0]["k"]),
            "selected_silhouette": float(scores[scores["k"].eq(selected_k)]["silhouette_cosine"].iloc[0]),
            "selected_davies_bouldin": float(scores[scores["k"].eq(selected_k)]["davies_bouldin"].iloc[0]),
            "selected_score": float(scores[scores["k"].eq(selected_k)]["selection_score"].iloc[0]),
        })
        summary_rows.append(summary)
        mood_rows.append(moodboards)
        algo_rows.extend(algorithm_scores(category, X, selected_k, labels))

    scores_all = pd.concat(all_scores, ignore_index=True)
    selected = pd.DataFrame(selected_rows)
    summary_all = pd.concat(summary_rows, ignore_index=True)
    moodboards_all = pd.concat(mood_rows, ignore_index=True)
    algorithms = pd.DataFrame(algo_rows)

    if cards_category is not None and main_figures_only:
        print(f"Saving cluster cards for {cards_category}...", flush=True)
        figure_cluster_cards(cards_category, summary_all, moodboards_all)
        print("Selected k values:")
        print(selected.to_string(index=False))
        print(f"Saved {cards_category} cluster cards to {OUT.relative_to(ROOT)}")
        return

    scores_all.to_csv(OUT / "cluster_k_selection_scores.csv", index=False)
    selected.to_csv(OUT / "cluster_k_selected.csv", index=False)
    summary_all.to_csv(OUT / "cluster_summary_by_category.csv", index=False)
    moodboards_all.to_csv(OUT / "cluster_representative_moodboards.csv", index=False)
    algorithms.to_csv(OUT / "cluster_algorithm_scores.csv", index=False)

    figure_k_selection(scores_all, selected)
    print("Saved cluster-selection figure.", flush=True)
    figure_cluster_maps(results)
    print("Saved cluster maps.", flush=True)
    figure_size_summary(summary_all)
    print("Saved cluster-size summary.", flush=True)
    figure_main_cluster_card_examples(summary_all, moodboards_all)
    print("Saved main cluster-card examples.", flush=True)
    if cards_category is not None:
        print(f"Saving cluster cards for {cards_category}...", flush=True)
        figure_cluster_cards(cards_category, summary_all, moodboards_all)
    elif not main_figures_only:
        for category in CATEGORIES:
            print(f"Saving cluster cards for {category}...", flush=True)
            figure_cluster_cards(category, summary_all, moodboards_all)

    print("Selected k values:")
    print(selected.to_string(index=False))
    print(f"Saved clustering figures to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-figures-only", action="store_true")
    parser.add_argument("--cards-category", choices=CATEGORIES)
    args, _ = parser.parse_known_args()
    main(main_figures_only=args.main_figures_only, cards_category=args.cards_category)
