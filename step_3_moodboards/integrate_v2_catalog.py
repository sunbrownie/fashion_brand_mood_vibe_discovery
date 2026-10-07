#!/usr/bin/env python3
"""Fully integrate approved V2 catalog additions into data, embeddings and clusters."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from build_generation_queue import SPECIAL_PROMPT_OVERRIDES


ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "recommender" / "hosting_bundle"
DATA = HOST / "data"
APP_ASSETS = ROOT / "ios" / "BrandMoodboardFinder" / "BrandMoodboardFinder" / "BrandMoodboards"
OVERLAY = APP_ASSETS / "v2_catalog_additions.json"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CATEGORY_ORDER = {name: index for index, name in enumerate(("clothes", "shoes", "bags", "jewellery", "swimwear_lingerie"))}

EXPANSION_PROMPTS = {
    ("shoes", "New Balance"): (
        "New Balance footwear balances grey heritage running, technical silver runners, sculptural lifestyle sneakers, "
        "retro court shoes, colourful racing shoes and rugged trail shoes."
    ),
    ("shoes", "Nike"): (
        "Nike footwear spans visible-air lifestyle shoes, cushioned road running, retro court icons, basketball, "
        "rugged ACG trail shoes and experimental sport design."
    ),
    ("shoes", "Puma"): (
        "Puma footwear combines motorsport driving shoes, terrace classics, suede icons, sculptural fashion sneakers, "
        "performance running and football boots."
    ),
    ("shoes", "Burton"): (
        "Burton snowboard boots combine high-support technical cuffs, BOA and traditional-lace closures, Step On "
        "compatibility, grippy soles and expressive snow-sport colour."
    ),
}

CATEGORY_MATERIALS = {
    "clothes": "cotton, wool, denim, leather, knit and technical seasonal fabrics",
    "shoes": "leather, suede, canvas, mesh, rubber and technical textiles",
    "bags": "leather, suede, canvas, nylon, woven fibres and metal hardware",
    "jewellery": "gold, silver, enamel, pearls, crystals and gemstones",
    "swimwear_lingerie": "stretch jersey, technical swim fabric, lace and soft elastic",
}

CLUSTER_OVERRIDES = {
    ("shoes", "new balance"): 7,
    ("shoes", "nike"): 7,
    ("shoes", "puma"): 7,
    ("shoes", "burton"): 11,
}

INVALID_TASTE_ROWS = {
    ("clothes", name)
    for name in {
        "acacia", "adriana degreas", "agua bendita", "andrea iyamah", "araks", "bather", "bluebella",
        "bondi born", "carine gilson", "cdlp", "cou cou intimates", "fleur du mal", "for love and lemons",
        "frankies bikinis", "fruity booty", "hunza g", "jade swim", "kiki de montparnasse", "le petit trou",
        "left on friday", "maison close", "marysia", "matteau", "maygel coronel", "melissa odabash", "mikoh",
        "montce", "natori", "only hearts", "orlebar brown", "peony swimwear", "saxx", "skims",
        "solid and striped", "suboo", "wild lovers",
    }
}


def normalised(value: str) -> str:
    folded = "".join(
        character
        for character in unicodedata.normalize("NFKD", str(value).replace("&", " and "))
        if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[a-z0-9]+", folded.lower()))


def slugify(value: str) -> str:
    return normalised(value).replace(" ", "_")


def sentence_matching(prompt: str, prefix: str) -> str:
    for sentence in re.split(r"(?<=[.!?])\s+", prompt.strip()):
        if sentence.lower().startswith(prefix.lower()):
            return sentence.rstrip(".")
    return ""


def descriptors(entry: dict) -> dict[str, str]:
    key = (entry["category"], entry["brand_name"])
    prompt = EXPANSION_PROMPTS.get(key) or SPECIAL_PROMPT_OVERRIDES.get(key)
    if not prompt:
        raise RuntimeError(f"Missing integration description for {key}")
    first = re.split(r"(?<=[.!?])\s+", prompt.strip())[0]
    first = re.sub(r"^Create a vertical six-panel\s+", "", first, flags=re.I)
    first = re.sub(r"\s+moodboard\s*", " ", first, flags=re.I).strip(" .")
    silhouettes = sentence_matching(prompt, "Show ")
    silhouettes = re.sub(r"^Show\s+(six|eight)\s+distinct\s+", "", silhouettes, flags=re.I)
    palette = sentence_matching(prompt, "Use ")
    palette = re.sub(r"^Use\s+", "", palette, flags=re.I)
    return {
        "aesthetic_keywords": first,
        "silhouettes": silhouettes or first,
        "materials": CATEGORY_MATERIALS[entry["category"]],
        "palette": palette or "brand-signature neutrals and seasonal colour accents",
    }


def build_catalog() -> tuple[pd.DataFrame, pd.DataFrame]:
    host_path = DATA / "brand_metadata.csv"
    host = pd.read_csv(host_path)
    overlay = json.loads(OVERLAY.read_text(encoding="utf-8"))["entries"]
    existing = {(row.category, normalised(row.brand_name)) for row in host.itertuples()}
    additions = []
    for entry in overlay:
        key = (entry["category"], normalised(entry["brand_name"]))
        if key in existing:
            continue
        fields = descriptors(entry)
        slug = slugify(entry["brand_name"])
        source = APP_ASSETS / entry["moodboard_asset_name"]
        target = DATA / "moodboards" / entry["category"] / f"{slug}.jpg"
        if not source.exists():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        additions.append({
            "brand_name": entry["brand_name"],
            "category": entry["category"],
            "official_website": entry["official_website"],
            "male": float(entry["male"]),
            "female": float(entry["female"]),
            "children": 0.0,
            "price": entry["price"],
            "moodboard_path": f"moodboards/{entry['category']}/{slug}.jpg",
            "slug": slug,
            "popularity": int(entry["popularity"]),
            "male_moodboard_path": "",
            **fields,
        })
        existing.add(key)
    host = pd.concat([host, pd.DataFrame(additions)], ignore_index=True, sort=False)
    host["_category_order"] = host["category"].map(CATEGORY_ORDER)
    host["_brand_order"] = host["brand_name"].astype(str).str.casefold()
    host = host.sort_values(["_category_order", "_brand_order"]).drop(columns=["_category_order", "_brand_order"])
    host = host.reset_index(drop=True)
    host.to_csv(host_path, index=False)

    prompt = (
        "Aesthetic: " + host["aesthetic_keywords"].fillna("").astype(str) + ". "
        "Silhouettes: " + host["silhouettes"].fillna("").astype(str) + ". "
        "Materials: " + host["materials"].fillna("").astype(str) + ". "
        "Palette: " + host["palette"].fillna("").astype(str) + "."
    )
    embedding_metadata = host[[
        "brand_name", "category", "official_website", "aesthetic_keywords", "silhouettes", "materials", "palette"
    ]].copy()
    embedding_metadata["prompt"] = prompt
    embedding_metadata["male"] = host["male"]
    embedding_metadata["female"] = host["female"]
    embedding_metadata["children"] = host["children"]
    embedding_metadata["price"] = host["price"]
    for path in (
        ROOT / "step_2_text_embeddings" / "brand_metadata.csv",
        ROOT / "step_2_text_embeddings" / "brand_metadata_today.csv",
    ):
        embedding_metadata.to_csv(path, index=False)
    return host, embedding_metadata


def update_source_csvs(host: pd.DataFrame) -> None:
    file_map = {
        "clothes": ROOT / "final_dataset" / "brands_clothes.csv",
        "shoes": ROOT / "final_dataset" / "brands_shoes.csv",
        "bags": ROOT / "final_dataset" / "brands_bags.csv",
        "jewellery": ROOT / "final_dataset" / "brands_jewellery.csv",
    }
    overlay = json.loads(OVERLAY.read_text(encoding="utf-8"))["entries"]
    overlay_keys = {(entry["category"], normalised(entry["brand_name"])) for entry in overlay}
    host_lookup = {(row.category, normalised(row.brand_name)): row for row in host.itertuples()}
    for category, path in file_map.items():
        frame = pd.read_csv(path)
        existing = {normalised(value) for value in frame["brand_name"]}
        rows = []
        for overlay_category, brand_key in sorted(overlay_keys):
            if overlay_category != category or brand_key in existing:
                continue
            row = host_lookup[(category, brand_key)]
            values = {
                "brand_name": row.brand_name,
                "official_website": row.official_website,
                "aesthetic_keywords": row.aesthetic_keywords,
                "silhouettes": row.silhouettes,
                "materials": row.materials,
                "palette": row.palette,
                "moodboard": f"{row.slug}.jpg",
                "male": int(row.male),
                "female": int(row.female),
                "price": row.price,
                "swimwear": 0,
                "lingerie": 0,
                "children": 0,
                "popularity": int(row.popularity),
            }
            rows.append({column: values.get(column, "") for column in frame.columns})
            existing.add(brand_key)
        if rows:
            frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
        frame = frame.sort_values("brand_name", key=lambda series: series.astype(str).str.casefold()).reset_index(drop=True)
        frame.to_csv(path, index=False)


def encode_text(metadata: pd.DataFrame) -> np.ndarray:
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        metadata["prompt"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    target = ROOT / "step_2_text_embeddings" / "brand_embeddings.npz"
    np.savez_compressed(target, embeddings=embeddings)
    shutil.copy2(target, DATA / "brand_embeddings.npz")
    return embeddings


def key_frame(frame: pd.DataFrame) -> dict[tuple[str, str], int]:
    return {
        (str(row.category), normalised(row.brand_name)): index
        for index, row in frame.iterrows()
    }


def update_projection_and_clusters(host: pd.DataFrame, embeddings: np.ndarray) -> None:
    taste_mask = pd.Series(
        [
            (row.category, normalised(row.brand_name)) not in INVALID_TASTE_ROWS
            for row in host.itertuples()
        ],
        index=host.index,
    )
    host = host.loc[taste_mask].reset_index(drop=True)
    embeddings = embeddings[taste_mask.to_numpy()]

    old_assignments = pd.read_csv(DATA / "ios_brand_cluster_assignments.csv")
    summary = pd.read_csv(DATA / "ios_cluster_summary.csv")
    metadata_lookup = key_frame(host)
    old_lookup = {
        (str(row.category), normalised(row.brand_name)): int(row.cluster)
        for row in old_assignments.itertuples()
    }
    summary_lookup = {
        (str(row.category), int(row.cluster)): row
        for row in summary.itertuples()
    }

    centroids: dict[tuple[str, int], np.ndarray] = {}
    for category in host["category"].unique():
        clusters = sorted(summary[summary["category"].eq(category)]["cluster"].astype(int).unique())
        for cluster in clusters:
            indices = [
                metadata_lookup[key]
                for key, value in old_lookup.items()
                if key[0] == category and value == cluster and key in metadata_lookup
            ]
            if not indices:
                raise RuntimeError(f"No centroid members for {category} cluster {cluster}")
            centroid = embeddings[indices].mean(axis=0)
            centroids[(category, cluster)] = centroid / max(np.linalg.norm(centroid), 1e-12)

    assignments = []
    assignment_values: dict[tuple[str, str], int] = {}
    for row_index, row in host.iterrows():
        key = (row["category"], normalised(row["brand_name"]))
        cluster = CLUSTER_OVERRIDES.get(key, old_lookup.get(key))
        if cluster is None:
            candidates = [(cid, centroid) for (category, cid), centroid in centroids.items() if category == row["category"]]
            scores = [(cid, float(embeddings[row_index] @ centroid)) for cid, centroid in candidates]
            cluster = max(scores, key=lambda item: item[1])[0]
        assignment_values[key] = cluster
        info = summary_lookup[(row["category"], cluster)]
        assignments.append({
            "brand_name": row["brand_name"],
            "category": row["category"],
            "cluster": cluster,
            "cluster_id": f"{row['category']}-{cluster}",
            "cluster_title": info.cluster_title,
            "top_aesthetic_terms": info.top_aesthetic_terms,
            "supporting_terms": info.supporting_terms,
            "example_brands": info.example_brands,
            "representative_brand_name": str(info.example_brands).split(",")[0].strip(),
            "representative_moodboard_path": f"moodboards/{row['category']}/{slugify(str(info.example_brands).split(',')[0].strip())}.jpg",
            "selected_k": (
                int(info.selected_k)
                if pd.notna(info.selected_k)
                else int(summary[summary["category"].eq(row["category"])]["cluster"].nunique())
            ),
        })
    assignment_frame = pd.DataFrame(assignments)
    assignment_frame.to_csv(DATA / "ios_brand_cluster_assignments.csv", index=False)
    assignment_frame.to_csv(DATA / "ios_brand_cluster_assignments.tsv", index=False, sep="\t")

    pca_values = PCA(n_components=48, random_state=42).fit_transform(embeddings)
    pca_frame = host[["brand_name", "category"]].copy()
    for index in range(48):
        pca_frame[f"e{index}"] = pca_values[:, index]
    pca_frame.to_csv(DATA / "ios_brand_embeddings_pca48.csv", index=False, float_format="%.6f")

    import umap

    umap_parts = []
    for category, category_rows in host.groupby("category", sort=False):
        indices = category_rows.index.to_numpy()
        raw = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, len(indices) - 1),
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        ).fit_transform(embeddings[indices])
        mins = raw.min(axis=0)
        spans = np.maximum(raw.max(axis=0) - mins, 1e-12)
        scaled = (raw - mins) / spans
        part = category_rows[["brand_name", "category"]].copy()
        part["cluster"] = [assignment_values[(category, normalised(name))] for name in part["brand_name"]]
        part["cluster_id"] = [f"{category}-{value}" for value in part["cluster"]]
        part["cluster_title"] = [summary_lookup[(category, int(value))].cluster_title for value in part["cluster"]]
        part["x"], part["y"] = scaled[:, 0], scaled[:, 1]
        part["umap_x"], part["umap_y"] = raw[:, 0], raw[:, 1]
        umap_parts.append(part)
    umap_frame = pd.concat(umap_parts, ignore_index=True)
    umap_frame.to_csv(DATA / "ios_brand_umap.csv", index=False, float_format="%.8f")
    umap_frame.to_csv(DATA / "ios_brand_umap.tsv", index=False, sep="\t", float_format="%.8f")

    updated_summary = summary.copy()
    for index, row in updated_summary.iterrows():
        members = assignment_frame[
            assignment_frame["category"].eq(row["category"])
            & assignment_frame["cluster"].eq(int(row["cluster"]))
        ]
        updated_summary.at[index, "n_brands"] = len(members)
    updated_summary["n_brands"] = updated_summary["n_brands"].astype(int)
    updated_summary.to_csv(DATA / "ios_cluster_summary.csv", index=False)


def copy_cluster_exports_to_ios() -> None:
    target = ROOT / "ios" / "BrandMoodboardFinder" / "BrandMoodboardFinder" / "ClusterData"
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "ios_brand_cluster_assignments.csv",
        "ios_brand_cluster_assignments.tsv",
        "ios_brand_embeddings_pca48.csv",
        "ios_brand_umap.csv",
        "ios_brand_umap.tsv",
        "ios_cluster_summary.csv",
    ):
        shutil.copy2(DATA / name, target / name)


def validate(host: pd.DataFrame, embeddings: np.ndarray) -> None:
    full_expected = len(host)
    expected = sum(
        (row.category, normalised(row.brand_name)) not in INVALID_TASTE_ROWS
        for row in host.itertuples()
    )
    files = {
        "PCA": DATA / "ios_brand_embeddings_pca48.csv",
        "clusters": DATA / "ios_brand_cluster_assignments.csv",
        "UMAP": DATA / "ios_brand_umap.csv",
    }
    if embeddings.shape != (full_expected, 768):
        raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")
    for label, path in files.items():
        rows = len(pd.read_csv(path))
        if rows != expected:
            raise RuntimeError(f"{label} has {rows} rows; expected {expected}")
    if int(pd.read_csv(DATA / "ios_cluster_summary.csv")["n_brands"].sum()) != expected:
        raise RuntimeError("Cluster summary counts do not match the integrated catalog")
    overlay = json.loads(OVERLAY.read_text(encoding="utf-8"))["entries"]
    keys = {(row.category, normalised(row.brand_name)) for row in host.itertuples()}
    missing = [entry["review_id"] for entry in overlay if (entry["category"], normalised(entry["brand_name"])) not in keys]
    if missing:
        raise RuntimeError(f"Integrated catalog is missing overlay entries: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help="Reuse the already generated 768-D matrix and rebuild only projections and cluster exports.",
    )
    args = parser.parse_args()
    host, embedding_metadata = build_catalog()
    update_source_csvs(host)
    if args.reuse_embeddings:
        embeddings = np.load(ROOT / "step_2_text_embeddings" / "brand_embeddings.npz")["embeddings"].astype("float32")
        shutil.copy2(ROOT / "step_2_text_embeddings" / "brand_embeddings.npz", DATA / "brand_embeddings.npz")
    else:
        embeddings = encode_text(embedding_metadata)
    update_projection_and_clusters(host, embeddings)
    copy_cluster_exports_to_ios()
    validate(host, embeddings)
    taste_rows = len(pd.read_csv(DATA / "ios_brand_cluster_assignments.csv"))
    print(
        f"Integrated {len(host)} brand-category rows with {embeddings.shape[1]}-D embeddings and "
        f"{taste_rows} valid taste-space rows in stable named clusters."
    )


if __name__ == "__main__":
    main()
