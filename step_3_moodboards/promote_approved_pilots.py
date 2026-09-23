#!/usr/bin/env python3
"""Promote self-reviewed candidates into reversible V2 analysis and iOS assets."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CORRECTED_ROOT = ROOT / "moodboards_creation" / "moodboards_corrected" / "v2"
PILOT_MANIFEST = CORRECTED_ROOT / "pilot_candidates.json"
IOS_ASSET_ROOT = (
    ROOT
    / "ios"
    / "BrandMoodboardFinder"
    / "BrandMoodboardFinder"
    / "BrandMoodboards"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug_from_candidate(path: Path) -> str:
    return path.stem.split("__candidate_", maxsplit=1)[0]


def main() -> None:
    payload = json.loads(PILOT_MANIFEST.read_text(encoding="utf-8"))
    approved = [
        item for item in payload["candidates"]
        if item.get("candidate_status") == "self_review_approved"
    ]
    if not approved:
        raise SystemExit("No self-review-approved candidates found.")

    ios_entries = []
    for item in approved:
        candidate = CORRECTED_ROOT / item["path"]
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        if sha256(candidate) != item["sha256"]:
            raise RuntimeError(f"Candidate checksum mismatch: {candidate}")

        category = item["category"]
        slug = slug_from_candidate(candidate)
        canonical = CORRECTED_ROOT / category / f"{slug}.png"
        ios_path = IOS_ASSET_ROOT / f"v2__{category}__{slug}.jpg"
        if canonical.exists() or ios_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite an existing V2 asset: {canonical} or {ios_path}"
            )

        shutil.copy2(candidate, canonical)
        with Image.open(candidate) as image:
            image.convert("RGB").save(ios_path, format="JPEG", quality=92, optimize=True)

        ios_entries.append({
            "review_id": item["review_id"],
            "category": category,
            "brand_name": item["brand_name"],
            "slug": slug,
            "candidate_path": str(candidate.relative_to(ROOT)),
            "analysis_v2_path": str(canonical.relative_to(ROOT)),
            "ios_v2_path": str(ios_path.relative_to(ROOT / "ios" / "BrandMoodboardFinder")),
            "source_sha256": item["sha256"],
            "ios_sha256": sha256(ios_path),
            "rollback": "Remove the v2__ asset or disable the V2 lookup; the original V1 asset is unchanged.",
        })

    ios_manifest = IOS_ASSET_ROOT / "v2_manifest.json"
    ios_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "policy": "V2 assets are preferred when present. V1 assets remain unchanged for rollback.",
                "entries": ios_entries,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Promoted {len(ios_entries)} approved V2 assets without changing V1 files.")


if __name__ == "__main__":
    main()
