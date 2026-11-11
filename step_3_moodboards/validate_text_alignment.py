#!/usr/bin/env python3
"""Validate reviewed text metadata, MPNet embeddings and downstream app exports."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "recommender" / "hosting_bundle" / "data"
STEP2 = ROOT / "step_2_text_embeddings"
IOS = ROOT / "ios" / "BrandMoodboardFinder" / "BrandMoodboardFinder"
REVIEW = ROOT / "step_3_moodboards" / "review"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
VIBE_COLUMNS = ("aesthetic_keywords", "silhouettes", "materials", "palette")
EXPECTED_CATALOG_ROWS = 1799
EXPECTED_TASTE_ROWS = 1799
EXPECTED_OVERRIDE_ROWS = 77
EXPECTED_V2_MANIFEST_ROWS = 383


def normalised(value: object) -> str:
    folded = "".join(
        character
        for character in unicodedata.normalize("NFKD", str(value).replace("&", " and "))
        if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[a-z0-9]+", folded.lower()))


def key(row: object) -> tuple[str, str]:
    return str(row.category), normalised(row.brand_name)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    host = pd.read_csv(DATA / "brand_metadata.csv").fillna("")
    metadata = pd.read_csv(STEP2 / "brand_metadata.csv").fillna("")
    metadata_today = pd.read_csv(STEP2 / "brand_metadata_today.csv").fillna("")
    host_keys = [key(row) for row in host.itertuples(index=False)]
    metadata_keys = [key(row) for row in metadata.itertuples(index=False)]
    if len(host) != EXPECTED_CATALOG_ROWS or len(set(host_keys)) != EXPECTED_CATALOG_ROWS:
        raise RuntimeError(f"Host metadata must contain {EXPECTED_CATALOG_ROWS:,} unique category-brand rows")
    if host_keys != metadata_keys:
        raise RuntimeError("Host and embedding metadata row order differs")
    if not metadata.equals(metadata_today):
        raise RuntimeError("Current and today embedding metadata differ")

    expected_prompt = (
        "Aesthetic: " + host["aesthetic_keywords"].astype(str) + ". "
        "Silhouettes: " + host["silhouettes"].astype(str) + ". "
        "Materials: " + host["materials"].astype(str) + ". "
        "Palette: " + host["palette"].astype(str) + "."
    )
    if not metadata["prompt"].astype(str).equals(expected_prompt):
        raise RuntimeError("Embedding prompts do not match the four host text fields")

    archive_path = STEP2 / "brand_embeddings.npz"
    archive = np.load(archive_path, allow_pickle=False)
    if set(archive.files) != {"embeddings", "model_name"}:
        raise RuntimeError(f"Unexpected NPZ keys: {archive.files}")
    embeddings = archive["embeddings"]
    if embeddings.shape != (EXPECTED_CATALOG_ROWS, 768) or embeddings.dtype != np.float32:
        raise RuntimeError(f"Unexpected embedding matrix: {embeddings.shape} {embeddings.dtype}")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Text embeddings are not L2-normalized")
    if str(archive["model_name"][0]) != MODEL_NAME:
        raise RuntimeError("Embedding model provenance is missing or incorrect")
    if sha256(archive_path) != sha256(DATA / "brand_embeddings.npz"):
        raise RuntimeError("Step-2 and hosting embedding archives differ")

    override_rows = list(csv.DictReader((REVIEW / "v2_text_alignment_overrides.csv").open()))
    if len(override_rows) != EXPECTED_OVERRIDE_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_OVERRIDE_ROWS} reviewed overrides, found {len(override_rows)}")
    host_lookup = {key(row): row for row in host.itertuples(index=False)}
    for row in override_rows:
        row_key = (row["category"], normalised(row["brand_name"]))
        if row_key not in host_lookup:
            raise RuntimeError(f"Override does not match the host catalog: {row_key}")
        host_row = host_lookup[row_key]
        for column in VIBE_COLUMNS:
            if str(getattr(host_row, column)) != row[column]:
                raise RuntimeError(f"Override was not applied: {row_key} {column}")

    manifest = json.loads((IOS / "BrandMoodboards" / "v2_manifest.json").read_text())["entries"]
    manifest_keys = {(entry["category"], normalised(entry["brand_name"])) for entry in manifest}
    if len(manifest) != EXPECTED_V2_MANIFEST_ROWS or not manifest_keys.issubset(set(host_keys)):
        raise RuntimeError("The V2 manifest is not fully represented in metadata")
    review = pd.read_csv(REVIEW / "v2_text_alignment_review.csv")
    if len(review) != 370 or int(review["text_review_status"].eq("updated_to_match_v2").sum()) != 63:
        raise RuntimeError("The complete V2 text-review register is inconsistent")

    generation_pattern = re.compile(r"product-only|\bmoodboard\b|one complete single", re.I)
    for row in host.itertuples(index=False):
        if generation_pattern.search(str(row.aesthetic_keywords)):
            raise RuntimeError(f"Generation instruction remains in aesthetic text: {key(row)}")
        if normalised(row.aesthetic_keywords) == normalised(row.silhouettes):
            raise RuntimeError(f"Aesthetic and silhouettes are duplicated: {key(row)}")

    category_files = {
        "clothes": "brands_clothes.csv",
        "shoes": "brands_shoes.csv",
        "bags": "brands_bags.csv",
        "jewellery": "brands_jewellery.csv",
        "swimwear_lingerie": "brands_swimwear_lingerie.csv",
    }
    source_lookup = {}
    for category, filename in category_files.items():
        frame = pd.read_csv(ROOT / "final_dataset" / filename).fillna("")
        source_lookup.update({(category, normalised(row.brand_name)): row for row in frame.itertuples(index=False)})
    for row in override_rows:
        row_key = (row["category"], normalised(row["brand_name"]))
        source = source_lookup[row_key]
        for column in VIBE_COLUMNS:
            if str(getattr(source, column)) != row[column]:
                raise RuntimeError(f"Source CSV was not updated: {row_key} {column}")

    exports = (
        "brand_public_cluster_assignments.csv",
        "ios_brand_cluster_assignments.csv",
        "ios_brand_cluster_assignments.tsv",
        "ios_brand_embeddings_pca48.csv",
        "ios_public_cluster_summary.csv",
        "ios_brand_umap.csv",
        "ios_brand_umap.tsv",
        "ios_cluster_summary.csv",
        "public_cluster_archetypes.csv",
    )
    for filename in exports:
        if sha256(DATA / filename) != sha256(IOS / "ClusterData" / filename):
            raise RuntimeError(f"Hosting and iOS exports differ: {filename}")
    for filename in ("ios_brand_cluster_assignments.csv", "ios_brand_embeddings_pca48.csv", "ios_brand_umap.csv"):
        if len(pd.read_csv(DATA / filename)) != EXPECTED_TASTE_ROWS:
            raise RuntimeError(f"Unexpected taste-space row count: {filename}")
    if int(pd.read_csv(DATA / "ios_cluster_summary.csv")["n_brands"].sum()) != EXPECTED_TASTE_ROWS:
        raise RuntimeError("Cluster summary does not reconcile to the taste-space rows")
    public_assignments = pd.read_csv(DATA / "ios_brand_cluster_assignments.csv")
    required_public = [
        "public_cluster_id", "public_cluster_description",
        "public_display_name", "public_display_description",
    ]
    if public_assignments[required_public].isna().any().any():
        raise RuntimeError("Customer-facing public archetype copy is incomplete")
    if len(pd.read_csv(DATA / "public_cluster_archetypes.csv")) != 15:
        raise RuntimeError("Expected exactly 15 customer-facing archetypes")
    if int(pd.read_csv(DATA / "ios_public_cluster_summary.csv")["n_brands"].sum()) != EXPECTED_TASTE_ROWS:
        raise RuntimeError("Public archetype summary does not reconcile to the taste-space rows")

    assignments = (IOS / "ClusterData" / "ios_brand_cluster_assignments.csv").read_text().rstrip("\n")
    generated = (IOS / "BuiltInTasteClusterAssignments.generated.swift").read_text()
    embedded = generated.split('static let csv = #"""\n', 1)[1].rsplit('\n"""#', 1)[0]
    if embedded != assignments:
        raise RuntimeError("The compiled Swift fallback is stale")

    print(
        f"Validated {EXPECTED_V2_MANIFEST_ROWS} reviewed moodboards, {EXPECTED_OVERRIDE_ROWS} text corrections, "
        f"{EXPECTED_CATALOG_ROWS:,} normalized MPNet vectors and {EXPECTED_TASTE_ROWS:,} synchronized iOS taste-space rows."
    )


if __name__ == "__main__":
    main()
