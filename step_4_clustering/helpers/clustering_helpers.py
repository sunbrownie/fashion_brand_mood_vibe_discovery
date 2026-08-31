# Helper code for the clustering notebook.

import sys

SHARED_HELPERS_DIR = ROOT / "step_4_clustering" / "helpers"
if str(SHARED_HELPERS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_HELPERS_DIR))
SOURCE_HELPERS_DIR = ROOT / "helpers"
if str(SOURCE_HELPERS_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_HELPERS_DIR))

from source_helpers import find_moodboard_file
from shared_cluster_helpers import (
    CLUSTER_TITLES,
    build_category_dataset,
    minmax,
    representative_brands,
    separate_text_labels,
    split_terms,
    tokenize_style_text,
)


def load_validation_families(path=None):
    """Takes the Table 3 source file and returns category family brand lists."""
    if path is None:
        candidates = [
            ROOT / "step_4_clustering" / "validation_families_thesis.json",
            ROOT / "validation_families_thesis.json",
            ROOT / "step_4_clustering" / "validation_families_thesis.csv",
            ROOT / "validation_families_thesis.csv",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])

    table = pd.read_json(path) if str(path).endswith(".json") else pd.read_csv(path)
    families = {}
    for row in table.itertuples(index=False):
        category = str(row.category).strip().lower()
        family = str(row.validation_family).strip()
        if isinstance(row.brands_included, list):
            brands = [str(brand).strip() for brand in row.brands_included if str(brand).strip()]
        else:
            brands = [brand.strip() for brand in str(row.brands_included).split(";") if brand.strip()]
        families.setdefault(category, {})[family] = brands
    return families


CATEGORY_VALIDATION_FAMILIES = load_validation_families()


CATEGORY_CLUSTER_CONFIG = {
    # k ranges used in the category sweeps.
    "clothes": {"k_range": range(7, 23)},
    "shoes": {"k_range": range(7, 26)},
    "bags": {"k_range": range(7, 23)},
    "jewellery": {"k_range": range(7, 23)},
}

K_SELECTION_WEIGHTS = {
    "silhouette": 0.45,
    "davies_bouldin": 0.35,
    "granularity": 0.20,
}

MOODBOARD_THUMBNAILS_PER_CLUSTER = 0
MAX_MOODBOARD_THUMBNAILS_ON_MAP = 0
ZOOM_MOODBOARDS_PER_CLUSTER = 6
RUN_ZOOMED_CLUSTER_PANELS = False

MOODBOARD_ROOT = ROOT / "moodboards_creation" / "moodboards"


def category_dataset(category):
    """Takes a category and returns its brand rows, number data, and name lookup."""
    if category not in CSV_FILES:
        raise ValueError(f"Unknown category {category!r}; choose one of {list(CSV_FILES)}")
    brands, X = build_category_dataset(source_rows, row_embeddings, category)
    name_to_idx = {name: i for i, name in enumerate(brands["brand_name"])}
    return brands, X, name_to_idx


def safe_silhouette(data, labels, metric="cosine"):
    """Takes data and labels and returns a cluster quality score."""
    labels = np.asarray(labels)
    non_noise = labels != -1
    n_clusters = len(set(labels) - {-1})
    if non_noise.sum() < 3 or n_clusters < 2:
        return np.nan
    if non_noise.sum() != len(labels):
        data = data[non_noise]
        labels = labels[non_noise]
    try:
        return silhouette_score(data, labels, metric=metric)
    except Exception:
        return np.nan


def valid_cluster_count(labels):
    """Takes labels and returns the number of clusters excluding noise."""
    return len(set(np.asarray(labels)) - {-1})


def compress_labels(labels):
    """Takes cluster labels and returns compact labels starting at zero."""
    labels = np.asarray(labels)
    mapping = {old: new for new, old in enumerate(sorted(set(labels)))}
    return np.array([mapping[x] for x in labels])


def project_2d(data, *, n_neighbors=30, min_dist=0.08, title="UMAP"):
    """Takes brand number data and returns 2D map positions."""
    if umap is None:
        raise ImportError("UMAP is required for cluster map visualisation.")
    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, max(2, len(data) - 1)),
        min_dist=min_dist,
        metric="cosine",
        random_state=RNG_SEED,
        spread=1.2,
    )
    return reducer.fit_transform(data), title

def cluster_descriptors(brands, labels, top_n=10):
    """Summarise each cluster using all embedded brand-card fields and brand names."""
    labels = np.asarray(labels)
    out = {}
    for label in sorted(set(labels)):
        if label == -1:
            continue
        rows = brands.loc[labels == label]

        aesthetic_counter = Counter()
        silhouette_counter = Counter()
        material_counter = Counter()
        palette_counter = Counter()
        all_word_counter = Counter()
        brand_tokens_counter = Counter()

        for _, row in rows.iterrows():
            brand_tokens_counter.update(tokenize_style_text(row["brand_name"]))
            for term in split_terms(row["aesthetic_keywords"]):
                tokens = tokenize_style_text(term)
                if tokens:
                    phrase = " ".join(tokens)
                    aesthetic_counter[phrase] += 1
                    all_word_counter.update(tokens)
            for term in split_terms(row["silhouettes"]):
                tokens = tokenize_style_text(term)
                if tokens:
                    silhouette_counter[" ".join(tokens)] += 1
                    all_word_counter.update(tokens)
            for term in split_terms(row["materials"]):
                tokens = tokenize_style_text(term)
                if tokens:
                    material_counter[" ".join(tokens)] += 1
                    all_word_counter.update(tokens)
            for term in split_terms(row["palette"]):
                tokens = tokenize_style_text(term)
                if tokens:
                    palette_counter[" ".join(tokens)] += 1
                    all_word_counter.update(tokens)

        top_aesthetic_terms = [term for term, _ in aesthetic_counter.most_common(top_n)]
        top_silhouette_terms = [term for term, _ in silhouette_counter.most_common(top_n)]
        top_material_terms = [term for term, _ in material_counter.most_common(top_n)]
        top_palette_terms = [term for term, _ in palette_counter.most_common(top_n)]
        top_evidence_terms = [term for term, _ in all_word_counter.most_common(top_n)]
        example_brand_names = rows["brand_name"].dropna().astype(str).head(12).tolist()

        if top_aesthetic_terms:
            headline = " / ".join(top_aesthetic_terms[:2])
        else:
            headline = "mixed aesthetic region"

        out[int(label)] = {
            "headline": headline,
            "vibes": [],
            "aesthetic_terms": top_aesthetic_terms,
            "silhouette_terms": top_silhouette_terms,
            "material_terms": top_material_terms,
            "palette_terms": top_palette_terms,
            "evidence_terms": top_evidence_terms,
            "brand_names": example_brand_names,
            "brand_tokens": [term for term, _ in brand_tokens_counter.most_common(8)],
            "plot_label": headline if len(headline) <= 42 else headline[:39] + "...",
        }
    return out


def make_distinct_cluster_titles(category, cluster_descriptions, reps):
    """Attach the final thesis title to each cluster."""
    for label, desc in cluster_descriptions.items():
        examples = reps.query("cluster == @label")["brand_name"].head(10).tolist() if not reps.empty else []
        desc["brand_names"] = list(dict.fromkeys(desc.get("brand_names", []) + examples))
        title = CLUSTER_TITLES.get((category, int(label)), f"Cluster {int(label)}")
        desc["cluster_title"] = title
        title = desc["cluster_title"]
        desc["plot_label"] = title if len(title) <= 34 else title[:31] + "..."
    return cluster_descriptions


def add_nearest_cluster_notes(cluster_descriptions, X, labels):
    """Takes cluster descriptions and adds nearest cluster notes."""
    labels = np.asarray(labels)
    centroids = {}
    for label in sorted(set(labels)):
        if label == -1:
            continue
        idx = np.where(labels == label)[0]
        centroids[int(label)] = normalize(X[idx].mean(axis=0, keepdims=True))[0]
    if len(centroids) < 2:
        return cluster_descriptions
    cluster_ids = list(centroids)
    C = np.vstack([centroids[cid] for cid in cluster_ids])
    sims = cosine_similarity(C)
    np.fill_diagonal(sims, -np.inf)
    for row_i, cid in enumerate(cluster_ids):
        order = np.argsort(-sims[row_i])[:2]
        nearest_info = []
        for near_i in order:
            if not np.isfinite(sims[row_i, near_i]):
                continue
            near_cid = cluster_ids[int(near_i)]
            nearest_info.append({
                "cluster": near_cid,
                "title": cluster_descriptions.get(near_cid, {}).get("cluster_title", f"cluster {near_cid}"),
                "similarity": float(sims[row_i, near_i]),
            })
        if nearest_info:
            cluster_descriptions[cid]["nearest_cluster"] = nearest_info[0]["cluster"]
            cluster_descriptions[cid]["nearest_cluster_title"] = nearest_info[0]["title"]
            cluster_descriptions[cid]["nearest_similarity"] = nearest_info[0]["similarity"]
        if len(nearest_info) > 1:
            cluster_descriptions[cid]["second_nearest_cluster"] = nearest_info[1]["cluster"]
            cluster_descriptions[cid]["second_nearest_cluster_title"] = nearest_info[1]["title"]
            cluster_descriptions[cid]["second_nearest_similarity"] = nearest_info[1]["similarity"]
        cluster_descriptions[cid]["nearest_clusters"] = nearest_info
    return cluster_descriptions

def validate_category(category, brands, X, name_to_idx, families):
    """Takes validation families and returns checks by comparing known brand groups."""
    records = []
    missing = []
    for family, names in families.items():
        present = [name for name in names if name in name_to_idx]
        if len(present) < 2:
            missing.extend([name for name in names if name not in name_to_idx])
            continue
        for name in present:
            records.append({"brand_name": name, "expected_family": family, "idx": name_to_idx[name]})

    validation = pd.DataFrame(records)
    if validation.empty or validation["expected_family"].nunique() < 2:
        print(f"No usable validation set for {category}.")
        return None

    X_val = X[validation["idx"].to_numpy()]
    val_dist = cosine_distances(X_val)

    print(f"Validation brands for {category}: {len(validation)} across {validation['expected_family'].nunique()} families")
    if missing:
        print("Missing validation brands:", sorted(set(missing)))
    display(validation[["brand_name", "expected_family"]].sort_values(["expected_family", "brand_name"]))

    pair_rows = []
    for i, j in combinations(range(len(validation)), 2):
        same = validation.loc[i, "expected_family"] == validation.loc[j, "expected_family"]
        pair_rows.append({"same_family": same, "cosine_distance": val_dist[i, j]})
    pair_df = pd.DataFrame(pair_rows)
    display(
        pair_df.groupby("same_family")["cosine_distance"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .rename(index={True: "same family", False: "different family"})
    )

    sim = cosine_similarity(X_val)
    np.fill_diagonal(sim, -np.inf)
    family_counts = validation["expected_family"].value_counts()
    nn_rows = []
    for i, row in validation.iterrows():
        family = row["expected_family"]
        k_i = min(3, int(family_counts[family]) - 1)
        nn = np.argsort(-sim[i])[:k_i]
        nn_rows.append({
            "brand_name": row["brand_name"],
            "expected_family": family,
            "k_i": k_i,
            "top_k_neighbours": ", ".join(validation.loc[nn, "brand_name"]),
            "family_precision_at_k_i": np.mean(validation.loc[nn, "expected_family"].to_numpy() == family),
        })
    nnk = pd.DataFrame(nn_rows)
    print("Mean validation precision@k_i:", round(nnk["family_precision_at_k_i"].mean(), 3))
    display(nnk.sort_values(["expected_family", "brand_name"]))

    y_val = LabelEncoder().fit_transform(validation["expected_family"])
    n_families = validation["expected_family"].nunique()
    methods = {
        f"kmeans_k{n_families}": KMeans(n_clusters=n_families, n_init=50, random_state=RNG_SEED).fit_predict(X_val),
        f"agglomerative_k{n_families}": AgglomerativeClustering(n_clusters=n_families, metric="cosine", linkage="average").fit_predict(X_val),
    }
    try:
        methods["sklearn.HDBSCAN"] = HDBSCAN(min_cluster_size=max(3, min(validation["expected_family"].value_counts())), min_samples=2, metric="cosine").fit_predict(X_val)
    except Exception as exc:
        print("Validation HDBSCAN skipped:", exc)

    score_rows = []
    assignments = validation[["brand_name", "expected_family"]].copy()
    for method, labels in methods.items():
        labels_for_metrics = compress_labels(labels) if -1 in labels else labels
        assignments[method] = labels
        score_rows.append({
            "method": method,
            "n_clusters_ex_noise": valid_cluster_count(labels),
            "noise_points": int(np.sum(np.asarray(labels) == -1)),
            "silhouette_cosine": safe_silhouette(X_val, labels),
            "ARI_vs_manual_families": adjusted_rand_score(y_val, labels_for_metrics),
            "NMI_vs_manual_families": normalized_mutual_info_score(
                y_val,
                labels_for_metrics,
                average_method="arithmetic",
            ),
        })
    display(pd.DataFrame(score_rows).sort_values("ARI_vs_manual_families", ascending=False))
    display(assignments.sort_values(["expected_family", "brand_name"]))
    return {"validation": validation, "X_val": X_val, "assignments": assignments}


def select_k_from_sweep(k_sweep):
    """Takes k sweep scores and returns the selected k plus scored rows."""
    scores = k_sweep.copy()

    scores["silhouette_norm"] = minmax(scores["silhouette_cosine"], higher_is_better=True)
    scores["davies_bouldin_norm"] = minmax(scores["davies_bouldin"], higher_is_better=False)
    scores["granularity_norm"] = minmax(scores["k"], higher_is_better=True)
    scores["selection_score"] = (
        K_SELECTION_WEIGHTS["silhouette"] * scores["silhouette_norm"]
        + K_SELECTION_WEIGHTS["davies_bouldin"] * scores["davies_bouldin_norm"]
        + K_SELECTION_WEIGHTS["granularity"] * scores["granularity_norm"]
    )

    # If scores are close, use the larger k.
    best_score = scores["selection_score"].max()
    near_best = scores[scores["selection_score"] >= best_score - 0.03]
    selected_row = near_best.sort_values(["k", "selection_score"], ascending=[False, False]).iloc[0]
    return int(selected_row["k"]), scores


def kmeans_sweep_rows(category, X, k_values, *, desc_suffix="k sweep", keep_labels=True):
    """Takes k values and returns k-means score rows, plus labels when requested."""
    k_rows = []
    labels_by_k = {}
    for k in tqdm(list(k_values), desc=f"{category}: k-means {desc_suffix}"):
        if k >= len(X):
            continue
        labels = KMeans(n_clusters=k, n_init=30, random_state=RNG_SEED).fit_predict(X)
        counts = np.bincount(labels)
        if keep_labels:
            labels_by_k[k] = labels
        k_rows.append({
            "method": "kmeans",
            "k": k,
            "silhouette_cosine": silhouette_score(X, labels, metric="cosine"),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "min_cluster_size": int(counts.min()),
            "clusters_lt_5": int((counts < 5).sum()),
        })
    return pd.DataFrame(k_rows), labels_by_k


def cluster_category(category, brands, X, k_range=None, target_k=None):
    """Takes category data and returns clusters by testing k values."""
    config = CATEGORY_CLUSTER_CONFIG.get(category, {})
    if k_range is None:
        k_range = config.get("k_range", range(7, 23))
    diagnostic_k_range = config.get("diagnostic_k_range")
    max_allowed_k = max(list(k_range))

    svd_dims = min(50, X.shape[1] - 1, len(X) - 1)
    X_svd = normalize(TruncatedSVD(n_components=svd_dims, random_state=RNG_SEED).fit_transform(X))

    k_sweep, labels_by_k = kmeans_sweep_rows(category, X, k_range, keep_labels=True)
    silhouette_best_k = int(k_sweep.sort_values("silhouette_cosine", ascending=False).iloc[0]["k"])
    db_best_k = int(k_sweep.sort_values("davies_bouldin", ascending=True).iloc[0]["k"])
    adaptive_k, scored_sweep = select_k_from_sweep(k_sweep)
    selected_k = int(target_k if target_k in labels_by_k else adaptive_k)
    main_labels = labels_by_k[selected_k]

    print(f"Best k by silhouette for {category}: {silhouette_best_k}")
    print(f"Best k by Davies-Bouldin for {category}: {db_best_k}")
    print(f"Adaptive selected k for granular map: {selected_k}")
    display(scored_sweep.sort_values("selection_score", ascending=False))

    diagnostic_sweep = pd.DataFrame()
    if diagnostic_k_range is not None:
        diagnostic_values = [k for k in diagnostic_k_range if k > max_allowed_k]
        if diagnostic_values:
            diagnostic_sweep, _ = kmeans_sweep_rows(
                category,
                X,
                diagnostic_values,
                desc_suffix="diagnostic boundary sweep",
                keep_labels=False,
            )
            if not diagnostic_sweep.empty:
                print(
                    f"Diagnostic-only k values beyond {max_allowed_k}. "
                    "These are not eligible for selection without a minimum-size or stability constraint."
                )
                display(diagnostic_sweep.sort_values("k"))

    fig, ax = plt.subplots(figsize=(14, 6))
    plot_scores = scored_sweep.melt(
        id_vars="k",
        value_vars=["silhouette_norm", "davies_bouldin_norm", "selection_score"],
        var_name="criterion",
        value_name="normalized_score",
    )
    criterion_labels = {
        "silhouette_norm": "silhouette (higher)",
        "davies_bouldin_norm": "Davies-Bouldin inverted (lower DB)",
        "selection_score": "combined selection score",
    }
    plot_scores["criterion"] = plot_scores["criterion"].map(criterion_labels)
    sns.lineplot(data=plot_scores, x="k", y="normalized_score", hue="criterion", marker="o", ax=ax)
    ax.axvline(silhouette_best_k, color="#264653", linestyle="--", alpha=0.65, label="silhouette best")
    ax.axvline(db_best_k, color="#2A9D8F", linestyle="-.", alpha=0.7, label="DB best")
    ax.axvline(selected_k, color="#E76F51", linestyle=":", linewidth=2.3, alpha=0.95, label="selected")
    ax.set_title("Adaptive k choice: overlap of silhouette and Davies-Bouldin, with granularity preference", fontsize=14, weight="bold")
    ax.set_xlabel("number of clusters (k)", fontsize=13)
    ax.set_ylabel("normalized score", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=10, loc="best")
    plt.tight_layout()
    plt.show()

    full_cluster_labels = {f"kmeans_k{selected_k}": main_labels}
    full_cluster_labels[f"agglomerative_k{selected_k}"] = AgglomerativeClustering(n_clusters=selected_k, metric="cosine", linkage="average").fit_predict(X)

    try:
        hdb = HDBSCAN(min_cluster_size=max(6, len(X) // 70), min_samples=max(3, len(X) // 180), metric="cosine").fit_predict(X_svd)
        hdb_n = valid_cluster_count(hdb)
        if 2 <= hdb_n <= max_allowed_k:
            full_cluster_labels["sklearn.HDBSCAN"] = hdb
        elif hdb_n > max_allowed_k:
            print(f"Full-space HDBSCAN produced {hdb_n} clusters, above max k={max_allowed_k}; skipping it in the comparison table.")
    except Exception as exc:
        print("Full-space HDBSCAN skipped:", exc)

    score_rows = []
    for method, labels in full_cluster_labels.items():
        data_for_score = X_svd if "HDBSCAN" in method else X
        score_rows.append({
            "method": method,
            "n_clusters_ex_noise": valid_cluster_count(labels),
            "noise_points": int(np.sum(np.asarray(labels) == -1)),
            "silhouette_cosine": safe_silhouette(data_for_score, labels),
        })
    display(pd.DataFrame(score_rows).sort_values("silhouette_cosine", ascending=False))
    return {
        "silhouette_best_k": silhouette_best_k,
        "davies_bouldin_best_k": db_best_k,
        "selected_k": selected_k,
        "k_sweep": scored_sweep,
        "diagnostic_k_sweep": diagnostic_sweep,
        "main_labels": main_labels,
        "all_labels": full_cluster_labels,
        "X_svd": X_svd,
    }


def phrase_list(values, n=4):
    """Takes values and returns a short comma list by dropping blanks."""
    cleaned = [str(v).strip() for v in values if str(v).strip() and str(v).strip().lower() not in {"nan", "none"}]
    return ", ".join(cleaned[:n])


def local_cluster_description(desc, examples):
    """Takes cluster terms and examples and returns one short description."""
    title = desc.get("cluster_title", desc.get("headline", "This region"))
    aesthetic = phrase_list(desc.get("aesthetic_terms", []), 4)
    silhouettes = phrase_list(desc.get("silhouette_terms", []), 3)
    materials = phrase_list(desc.get("material_terms", []), 2)
    palette = phrase_list(desc.get("palette_terms", []), 2)
    example_text = phrase_list(examples or desc.get("brand_names", []), 4)

    clauses = []
    if aesthetic:
        clauses.append(aesthetic)
    if silhouettes:
        clauses.append(f"silhouettes around {silhouettes}")
    if materials:
        clauses.append(f"materials such as {materials}")
    if palette:
        clauses.append(f"palette cues of {palette}")
    evidence = "; ".join(clauses) if clauses else "mixed brand-card signals"

    nearest = desc.get("nearest_cluster")
    nearest_title = desc.get("nearest_cluster_title")
    second = desc.get("second_nearest_cluster")
    second_title = desc.get("second_nearest_cluster_title")
    neighbour_text = ""
    if nearest is not None and nearest_title and second is not None and second_title:
        neighbour_text = f" Closest neighbouring clusters: {nearest} ({nearest_title}) and {second} ({second_title})."
    elif nearest is not None and nearest_title:
        neighbour_text = f" Closest neighbouring cluster: {nearest} ({nearest_title})."

    brand_text = f", with representative brands including {example_text}" if example_text else ""
    return f"The {title} cluster is characterised by {evidence}{brand_text}.{neighbour_text}"


def cluster_text_description(category, cluster_id, desc, brands_in_cluster, examples):
    """Takes cluster data and returns a text description from saved fields."""
    return local_cluster_description(desc, examples)


def moodboard_path_for_brand(category, brands, name):
    """Takes a brand name and returns its moodboard path from the brand table."""
    brand_moodboards = brands.set_index("brand_name")["moodboard"].fillna("").to_dict()
    return find_moodboard_file(MOODBOARD_ROOT, category, name, brand_moodboards.get(name, ""))


def moodboards_for_cluster(category, brands, reps, cluster_id, n=5):
    """Takes a cluster id and returns moodboard rows for representative brands."""
    rows = []
    cluster_reps = reps[reps["cluster"] == cluster_id]
    for _, rep in cluster_reps.iterrows():
        path = moodboard_path_for_brand(category, brands, rep["brand_name"])
        if path is not None:
            rows.append({**rep.to_dict(), "moodboard_path": path})
        if len(rows) >= n:
            break
    return pd.DataFrame(rows)


def label_all_cluster_points(ax, sub, fontsize=12):
    """Takes an axis and cluster points and adds readable brand labels."""
    texts = []
    x_min, x_max = sub["x"].min(), sub["x"].max()
    y_min, y_max = sub["y"].min(), sub["y"].max()
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    center_x, center_y = sub["x"].mean(), sub["y"].mean()

    # Spread initial labels around the outside of the cloud.
    ordered = sub.assign(_angle=np.arctan2(sub["y"] - center_y, sub["x"] - center_x)).sort_values("_angle")
    for rank, (_, row) in enumerate(ordered.iterrows()):
        angle = row["_angle"]
        radial_x = np.cos(angle)
        radial_y = np.sin(angle)
        # Start labels outside the point cloud.
        tx = row["x"] + radial_x * 0.20 * x_span
        ty = row["y"] + radial_y * 0.18 * y_span
        # Small stagger for dense arcs.
        tx += ((rank % 3) - 1) * 0.025 * x_span
        ty += ((rank % 4) - 1.5) * 0.018 * y_span
        texts.append(
            ax.text(
                tx,
                ty,
                row["brand_name"],
                fontsize=fontsize,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="#555555", linewidth=0.55, alpha=0.92, pad=2.4),
                zorder=5,
            )
        )

    separate_text_labels(ax, texts, pad=3.0, max_iter=180)


def plot_zoomed_cluster_panels(category, brands, plot_df, reps, cluster_descriptions, cluster_palette):
    """Takes cluster outputs and plots zoomed panels with labels and moodboards."""
    for cluster_id in sorted(plot_df["cluster"].astype(int).unique()):
        sub = plot_df[plot_df["cluster"] == str(cluster_id)].copy()
        if sub.empty:
            continue
        desc = cluster_descriptions.get(int(cluster_id), {"headline": "mixed region"})
        cluster_brands = brands[brands["brand_name"].isin(sub["brand_name"])]
        examples = reps.query("cluster == @cluster_id")["brand_name"].head(8).tolist()
        description = cluster_text_description(category, cluster_id, desc, cluster_brands, examples)
        mood_df = moodboards_for_cluster(category, brands, reps, cluster_id, n=ZOOM_MOODBOARDS_PER_CLUSTER)
        cluster_color = cluster_palette.get(str(cluster_id), "#2A9D8F")

        fig = plt.figure(figsize=(22, 14))
        gs = fig.add_gridspec(3, 4, width_ratios=[3.25, 1.1, 1.1, 1.1], height_ratios=[0.17, 1.0, 1.0], wspace=0.07, hspace=0.16)
        ax_desc = fig.add_subplot(gs[0, :])
        ax_map = fig.add_subplot(gs[1:, 0])
        mood_axes = [fig.add_subplot(gs[r, c]) for r in [1, 2] for c in [1, 2, 3]]

        ax_desc.axis("off")
        ax_desc.text(0.0, 0.94, f"Cluster {cluster_id}: {desc.get('cluster_title', desc['headline'])}", fontsize=24, weight="bold", ha="left", va="top")
        ax_desc.text(0.0, 0.42, description, fontsize=17, ha="left", va="top", wrap=True)

        ax_map.scatter(sub["x"], sub["y"], s=90, color=cluster_color, alpha=0.90, edgecolor="black", linewidth=0.5)
        label_font = 13 if len(sub) <= 45 else 11.5 if len(sub) <= 90 else 10
        label_all_cluster_points(ax_map, sub, fontsize=label_font)

        # Fit close to the cluster cloud with room for labels.
        x_pad = max((sub["x"].max() - sub["x"].min()) * 0.14, 0.20)
        y_pad = max((sub["y"].max() - sub["y"].min()) * 0.14, 0.20)
        ax_map.set_xlim(sub["x"].min() - x_pad, sub["x"].max() + x_pad)
        ax_map.set_ylim(sub["y"].min() - y_pad, sub["y"].max() + y_pad)
        ax_map.set_title(f"Zoomed UMAP region: {len(sub)} brands", fontsize=18, weight="bold")
        ax_map.set_xlabel("UMAP-1", fontsize=15)
        ax_map.set_ylabel("UMAP-2", fontsize=15)
        ax_map.tick_params(axis="both", labelsize=13)
        ax_map.grid(alpha=0.25)

        for ax in mood_axes:
            ax.axis("off")
        if mood_df.empty:
            mood_axes[2].text(0.5, 0.5, "No matched moodboards", ha="center", va="center", fontsize=14)
        else:
            for ax, (_, row) in zip(mood_axes, mood_df.head(6).iterrows()):
                ax.set_title(row["brand_name"], fontsize=15, weight="bold", pad=7)
                img = Image.open(row["moodboard_path"]).convert("RGB")
                ax.imshow(img)
                ax.set_aspect("auto")
                ax.axis("off")
        plt.tight_layout()
        plt.show()

def interpret_and_plot_category(category, brands, X, main_labels, validation_result=None, max_moodboard_clusters=4):
    """Takes clustering outputs and returns summaries by describing and plotting clusters."""
    cluster_descriptions = cluster_descriptors(brands, main_labels, top_n=8)
    reps = representative_brands(brands, X, main_labels, top_n=14)
    cluster_descriptions = make_distinct_cluster_titles(category, cluster_descriptions, reps)
    cluster_descriptions = add_nearest_cluster_notes(cluster_descriptions, X, main_labels)
    for label, desc in cluster_descriptions.items():
        cluster_mask = main_labels == label
        cluster_brands = brands.loc[cluster_mask]
        examples = reps.query("cluster == @label")["brand_name"].head(8).tolist()
        desc["cluster_description"] = cluster_text_description(category, label, desc, cluster_brands, examples)

    summary_rows = []
    for label, desc in cluster_descriptions.items():
        examples = reps.query("cluster == @label")["brand_name"].head(5).tolist()
        summary_rows.append({
            "cluster": label,
            "n_brands": int(np.sum(main_labels == label)),
            "cluster_title": desc["cluster_title"],
            "vibe_label": desc["headline"],
            "cluster_description": desc["cluster_description"],
            "nearest_cluster": desc.get("nearest_cluster"),
            "nearest_cluster_title": desc.get("nearest_cluster_title"),
            "second_nearest_cluster": desc.get("second_nearest_cluster"),
            "second_nearest_cluster_title": desc.get("second_nearest_cluster_title"),
            "top_aesthetic_terms": ", ".join(desc["aesthetic_terms"]),
            "supporting_terms": ", ".join(desc["evidence_terms"][:6]),
            "example_brands": ", ".join(examples),
        })
    cluster_summary = pd.DataFrame(summary_rows).sort_values("n_brands", ascending=False)
    display(cluster_summary)
    display(reps.head(120))

    coords, projection_name = project_2d(X, n_neighbors=35, min_dist=0.05, title="UMAP")
    plot_df = brands[["brand_name", "category"]].copy()
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]
    plot_df["cluster"] = main_labels.astype(str)
    plot_df["is_validation"] = False
    if validation_result is not None:
        validation = validation_result["validation"]
        plot_df["is_validation"] = plot_df["brand_name"].isin(validation["brand_name"])
        plot_df = plot_df.merge(validation[["brand_name", "expected_family"]], on="brand_name", how="left")

    cluster_ids = sorted(plot_df["cluster"].unique(), key=lambda x: int(x))
    colors = sns.color_palette("tab20", n_colors=max(20, len(cluster_ids)))
    cluster_palette = {cid: colors[i % len(colors)] for i, cid in enumerate(cluster_ids)}

    fig, ax = plt.subplots(figsize=(22, 16))
    sns.scatterplot(data=plot_df, x="x", y="y", hue="cluster", palette=cluster_palette, s=34, linewidth=0, alpha=0.62, ax=ax)
    val_plot = plot_df[plot_df["is_validation"]]
    if len(val_plot):
        ax.scatter(val_plot["x"], val_plot["y"], s=145, facecolors="none", edgecolors="black", linewidths=1.4, label="validation brands")
    texts = []
    for cluster_id, desc in cluster_descriptions.items():
        sub = plot_df[plot_df["cluster"] == str(cluster_id)]
        if len(sub) == 0:
            continue
        texts.append(ax.text(sub["x"].median(), sub["y"].median(), desc.get("plot_label", desc.get("cluster_title", desc["headline"])), fontsize=14, weight="bold", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.34", facecolor="white", edgecolor="0.25", alpha=0.9), zorder=5))
    ax.margins(0.08)
    separate_text_labels(ax, texts, pad=4.0, max_iter=220)
    ax.set_title(f"{category.title()} brand embeddings: true {projection_name}\ncolour: k-means clusters; labels: text-derived fashion-vibe cluster descriptions", fontsize=19, weight="bold")
    ax.set_xlabel(f"{projection_name}-1", fontsize=16)
    ax.set_ylabel(f"{projection_name}-2", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, title="cluster", fontsize=12, title_fontsize=13)
    plt.tight_layout()
    plt.show()

    moodboard_rows = []
    for _, row in reps.iterrows():
        path = moodboard_path_for_brand(category, brands, row["brand_name"])
        if path is not None:
            moodboard_rows.append({**row.to_dict(), "moodboard_path": path})
    moodboard_df = pd.DataFrame(moodboard_rows)
    print(f"Moodboards matched for {category}: {len(moodboard_df)} representative examples")
    display(moodboard_df.drop(columns=["moodboard_path"], errors="ignore").head(40))
    if RUN_ZOOMED_CLUSTER_PANELS:
        plot_zoomed_cluster_panels(category, brands, plot_df, reps, cluster_descriptions, cluster_palette)
    return {"cluster_summary": cluster_summary, "representatives": reps, "moodboards": moodboard_df, "plot_df": plot_df}


def run_category_analysis(category, *, k_range=None, target_k=None):
    """Takes a category and returns the full validation, clustering, and plot analysis."""
    print("=" * 90)
    print(f"CATEGORY: {category.upper()}")
    print("=" * 90)
    brands, X, name_to_idx = category_dataset(category)
    print(f"{category}: {len(brands):,} unique brands; vector matrix {X.shape}")
    display(brands.sample(min(8, len(brands)), random_state=RNG_SEED)[["brand_name", "aesthetic_keywords", "palette", "moodboard"]])
    validation_result = validate_category(category, brands, X, name_to_idx, CATEGORY_VALIDATION_FAMILIES.get(category, {}))
    cluster_result = cluster_category(category, brands, X, k_range=k_range, target_k=target_k)
    interpretation_result = interpret_and_plot_category(category, brands, X, cluster_result["main_labels"], validation_result)
    return {"brands": brands, "X": X, "validation": validation_result, "clusters": cluster_result, "interpretation": interpretation_result}
