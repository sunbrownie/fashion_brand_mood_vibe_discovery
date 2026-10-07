#!/usr/bin/env python3
"""Publish the reviewed moodboards and derived artifacts to the existing HF Space."""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


ROOT = Path(__file__).resolve().parents[1]
IOS_ROOT = ROOT / "ios" / "BrandMoodboardFinder" / "BrandMoodboardFinder"
HF_ROOT = ROOT / "recommender" / "hosting_bundle"
REPO_ID = "Zilorina/brand-moodboard-recommender"

DERIVED_ARTIFACTS = (
    "README.md",
    "app.py",
    "data/brand_metadata.csv",
    "data/brand_embeddings.npz",
    "data/ios_brand_cluster_assignments.csv",
    "data/ios_brand_cluster_assignments.tsv",
    "data/ios_brand_embeddings_pca48.csv",
    "data/ios_brand_umap.csv",
    "data/ios_brand_umap.tsv",
    "data/ios_cluster_summary.csv",
    "data/moodboard_image_embeddings_clip_vit_b32.npz",
    "data/moodboard_image_embeddings_metadata.csv",
)


def moodboard_paths() -> list[str]:
    manifest_path = IOS_ROOT / "BrandMoodboards" / "v2_manifest.json"
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))["entries"]
    if len(entries) != 322:
        raise ValueError(f"Expected 322 accepted V2 moodboards, found {len(entries)}")

    paths: set[str] = set()
    for entry in entries:
        category = entry["category"]
        paths.add(f"data/moodboards/{category}/{entry['slug']}.jpg")
    return sorted(paths)


def main() -> None:
    relative_paths = [*DERIVED_ARTIFACTS, *moodboard_paths()]
    missing = [path for path in relative_paths if not (HF_ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing publish inputs: {missing}")

    operations = [
        CommitOperationAdd(path_in_repo=path, path_or_fileobj=HF_ROOT / path)
        for path in relative_paths
    ]
    result = HfApi().create_commit(
        repo_id=REPO_ID,
        repo_type="space",
        operations=operations,
        commit_message="Integrate reviewed V2 moodboards, catalog and embeddings",
        commit_description=(
            "Publishes all 322 accepted V2 moodboards, including corrected existing "
            "brands and new catalog entries; refreshes reviewed descriptions, text "
            "embeddings, clusters, PCA and UMAP; and rebuilds separate CLIP "
            "moodboard-image embeddings with checksummed metadata."
        ),
    )
    print(result.commit_url)


if __name__ == "__main__":
    main()
