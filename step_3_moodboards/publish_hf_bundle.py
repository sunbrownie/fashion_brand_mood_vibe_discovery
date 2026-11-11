#!/usr/bin/env python3
"""Publish the reviewed moodboards and derived artifacts to the existing HF Space."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi


ROOT = Path(__file__).resolve().parents[1]
IOS_ROOT = ROOT / "ios" / "BrandMoodboardFinder" / "BrandMoodboardFinder"
HF_ROOT = ROOT / "recommender" / "hosting_bundle"
REPO_ID = "Zilorina/brand-moodboard-recommender"
OBSOLETE_NON_MOODBOARD_PATHS = (
    "data/moodboard_image_embeddings_clip_vit_b32.npz",
    "data/moodboard_image_embeddings_metadata.csv",
)

DERIVED_ARTIFACTS = (
    "README.md",
    "app.py",
    "data/brand_metadata.csv",
    "data/brand_embeddings.npz",
    "data/brand_public_cluster_assignments.csv",
    "data/ios_brand_cluster_assignments.csv",
    "data/ios_brand_cluster_assignments.tsv",
    "data/ios_brand_embeddings_pca48.csv",
    "data/ios_public_cluster_summary.csv",
    "data/ios_brand_umap.csv",
    "data/ios_brand_umap.tsv",
    "data/ios_cluster_summary.csv",
    "data/public_cluster_archetypes.csv",
)

CORRECTED_NON_V2_MOODBOARDS = (
    "data/moodboards/clothes/cernucci.jpg",
    "data/moodboards/jewellery/cernucci.jpg",
    "data/moodboards/shoes/labante_london.jpg",
    "data/moodboards/bags/melie_bianco.jpg",
)


def moodboard_paths() -> list[str]:
    manifest_path = IOS_ROOT / "BrandMoodboards" / "v2_manifest.json"
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))["entries"]
    if len(entries) != 383:
        raise ValueError(f"Expected 383 accepted V2 moodboards, found {len(entries)}")

    paths: set[str] = set()
    for entry in entries:
        category = entry["category"]
        paths.add(f"data/moodboards/{category}/{entry['slug']}.jpg")
    return sorted(paths)


def expected_catalog_moodboard_paths() -> set[str]:
    metadata_path = HF_ROOT / "data" / "brand_metadata.csv"
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    paths = {f"data/{row['moodboard_path']}" for row in rows if row.get("moodboard_path")}
    if len(rows) != 1798 or len(paths) != 1798:
        raise ValueError(
            f"Expected 1,798 catalog rows and unique moodboards, found {len(rows)} rows and {len(paths)} paths"
        )
    return paths


def clean_local_stale_moodboards(expected_paths: set[str]) -> list[str]:
    expected_files = {(HF_ROOT / path).resolve() for path in expected_paths}
    local_files = set((HF_ROOT / "data" / "moodboards").rglob("*.jpg"))
    stale_files = sorted(path for path in local_files if path.resolve() not in expected_files)
    for path in stale_files:
        path.unlink()
    for relative_path in OBSOLETE_NON_MOODBOARD_PATHS:
        path = HF_ROOT / relative_path
        if path.exists():
            path.unlink()
    return [path.relative_to(HF_ROOT).as_posix() for path in stale_files]


def main() -> None:
    expected_paths = expected_catalog_moodboard_paths()
    removed_local = clean_local_stale_moodboards(expected_paths)
    # Publish the reviewed V2 set plus explicitly corrected non-V2 boards.
    # Uploading the complete 1.36 GB catalogue exceeds the Space's 1 GB commit
    # limit even though almost all existing moodboards are byte-identical.
    relative_paths = [
        *DERIVED_ARTIFACTS,
        *moodboard_paths(),
        *CORRECTED_NON_V2_MOODBOARDS,
    ]
    missing = [path for path in relative_paths if not (HF_ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing publish inputs: {missing}")

    api = HfApi()
    existing_remote = set(api.list_repo_files(repo_id=REPO_ID, repo_type="space"))
    stale_remote = sorted(
        path
        for path in existing_remote
        if path.startswith("data/moodboards/") and path.endswith(".jpg") and path not in expected_paths
    )
    obsolete_remote = sorted(path for path in OBSOLETE_NON_MOODBOARD_PATHS if path in existing_remote)

    operations = [
        CommitOperationAdd(path_in_repo=path, path_or_fileobj=HF_ROOT / path)
        for path in relative_paths
    ]
    operations.extend(
        CommitOperationDelete(path_in_repo=path)
        for path in [*stale_remote, *obsolete_remote]
    )
    result = api.create_commit(
        repo_id=REPO_ID,
        repo_type="space",
        operations=operations,
        commit_message="Publish final reviewed moodboards and refreshed text clusters",
        commit_description=(
            "Publishes the final category-truth review, corrected canonical moodboards, refreshed "
            "MPNet text embeddings and technical assignments, and customer-facing category "
            "archetypes. Removes unsupported category extensions, superseded moodboards and "
            "unused CLIP artifacts so the Space keeps one current board per active row."
        ),
    )
    print(f"Removed {len(removed_local)} stale local moodboards and {len(stale_remote)} stale remote moodboards.")
    print(result.commit_url)


if __name__ == "__main__":
    main()
