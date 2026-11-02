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
)


def moodboard_paths() -> list[str]:
    manifest_path = IOS_ROOT / "BrandMoodboards" / "v2_manifest.json"
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))["entries"]
    if len(entries) != 370:
        raise ValueError(f"Expected 370 accepted V2 moodboards, found {len(entries)}")

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
        commit_message="Refine Celine CDG Chan Luu and Alohas boards",
        commit_description=(
            "Publishes all 370 accepted V2 moodboards with Celine shoes restored to "
            "round one and product-grounded refinements for Comme des Garcons shoes, "
            "Chan Luu bags and Alohas shoes. Text embeddings, clusters, PCA and UMAP "
            "are unchanged; the runtime recommendation path remains text-embedding based."
        ),
    )
    print(result.commit_url)


if __name__ == "__main__":
    main()
