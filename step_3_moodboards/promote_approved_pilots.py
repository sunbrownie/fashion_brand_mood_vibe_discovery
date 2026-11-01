#!/usr/bin/env python3
"""Promote the canonical latest Ellina moodboards into app assets."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
CORRECTED_ROOT = ROOT / "moodboards_creation" / "moodboards_corrected" / "v2"
REVIEW_MANIFEST = ROOT / "step_3_moodboards" / "review" / "ellina_correction_manifest.json"
IOS_PROJECT_ROOT = ROOT / "ios" / "BrandMoodboardFinder"
IOS_ASSET_ROOT = IOS_PROJECT_ROOT / "BrandMoodboardFinder" / "BrandMoodboards"
IMAGE_ACTIONS = {"targeted_regenerate", "full_regenerate", "catalog_add_and_generate"}
CATALOG_VERIFY_REGENERATE_IDS = {"ER-046", "ER-244"}
IOS_V2_SIZE = (1024, 1536)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slugify(value: str) -> str:
    import re
    import unicodedata

    normalized = unicodedata.normalize("NFKD", value.replace("&", " and "))
    ascii_value = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "_", ascii_value.lower()).strip("_")


def promote_image(source: Path, analysis_path: Path, ios_path: Path) -> None:
    """Export one canonical latest source to the analysis and iOS targets."""
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    ios_path.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != analysis_path.resolve():
        shutil.copy2(source, analysis_path)
    with Image.open(source) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        if rgb.size == IOS_V2_SIZE:
            exported = rgb
        else:
            # Preserve the whole approved collage while exporting a true phone-
            # portrait asset. A blurred edge-to-edge extension avoids white bars
            # and keeps every approved product tile visible and undistorted.
            exported = ImageOps.fit(
                rgb,
                IOS_V2_SIZE,
                method=Image.Resampling.LANCZOS,
            ).filter(ImageFilter.GaussianBlur(radius=32))
            foreground = ImageOps.contain(
                rgb,
                IOS_V2_SIZE,
                method=Image.Resampling.LANCZOS,
            )
            offset = (
                (IOS_V2_SIZE[0] - foreground.width) // 2,
                (IOS_V2_SIZE[1] - foreground.height) // 2,
            )
            exported.paste(foreground, offset)
        exported.save(ios_path, format="JPEG", quality=92, optimize=True)


def entry_for(
    *,
    record: dict,
    source: Path,
    slug: str,
    brand_name: str | None = None,
    variant: str | None = None,
) -> dict:
    category = record["category"]
    analysis_path = CORRECTED_ROOT / category / f"{slug}.png"
    ios_path = IOS_ASSET_ROOT / f"v2__{category}__{slug}.jpg"
    promote_image(source, analysis_path, ios_path)
    result = {
        "review_id": record["review_id"],
        "review_ids": [record["review_id"]],
        "category": category,
        "brand_name": brand_name or record["brand_name"],
        "slug": slug,
        "source_path": str(source.relative_to(ROOT)),
        "analysis_v2_path": str(analysis_path.relative_to(ROOT)),
        "ios_v2_path": str(ios_path.relative_to(IOS_PROJECT_ROOT)),
        "source_sha256": sha256(source),
        "ios_sha256": sha256(ios_path),
        "rollback": "Restore the prior canonical asset from Git history or disable this v2__ asset.",
    }
    if variant:
        result["variant"] = variant
    return result


def main() -> None:
    review_records = json.loads(REVIEW_MANIFEST.read_text(encoding="utf-8"))["records"]
    corrections = [
        item
        for item in review_records
        if item["recommended_action"] in IMAGE_ACTIONS
        or item["review_id"] in CATALOG_VERIFY_REGENERATE_IDS
    ]
    ios_entries: list[dict] = []
    entries_by_target: dict[tuple[str, str], dict] = {}
    for record in corrections:
        category = record["category"]
        slug = slugify(record["brand_name"])
        targets = [(slug, record["brand_name"], CORRECTED_ROOT / category / f"{slug}.png", None)]
        if record["review_id"] == "ER-180":
            targets = [
                (
                    "comme_des_garcons",
                    "Comme des Garçons",
                    CORRECTED_ROOT / "shoes" / "comme_des_garcons.png",
                    "avant_garde",
                ),
                (
                    "comme_des_garcons_play",
                    "Comme des Garçons PLAY",
                    CORRECTED_ROOT / "shoes" / "comme_des_garcons_play.png",
                    "play",
                ),
            ]

        for target_slug, target_brand, source, variant in targets:
            if not source.exists():
                raise FileNotFoundError(source)
            key = (category, target_slug)
            existing = entries_by_target.get(key)
            if existing:
                if existing["source_sha256"] != sha256(source):
                    raise RuntimeError(
                        f"Conflicting approved sources for V2 target: {category}/{target_slug}"
                    )
                existing["review_ids"].append(record["review_id"])
                continue
            entry = entry_for(
                record=record,
                source=source,
                slug=target_slug,
                brand_name=target_brand,
                variant=variant,
            )
            entries_by_target[key] = entry
            ios_entries.append(entry)

    ios_manifest = IOS_ASSET_ROOT / "v2_manifest.json"
    ios_manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "policy": (
                    "V2 assets are preferred when present. Each category/brand stores only the "
                    "canonical latest moodboard; earlier versions remain recoverable from Git history."
                ),
                "entries": ios_entries,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Promoted {len(corrections)} approved boards to {len(ios_entries)} V2 assets "
        "from canonical latest sources without changing V1 files."
    )


if __name__ == "__main__":
    main()
