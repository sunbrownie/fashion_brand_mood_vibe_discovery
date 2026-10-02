#!/usr/bin/env python3
"""Build the official-link and Google Images reference manifest for Ellina review pages."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import quote_plus

from PIL import Image, ImageOps


def normalized(value: str) -> str:
    folded = "".join(
        character
        for character in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(character)
    )
    return "_".join(re.findall(r"[a-z0-9]+", folded.lower().replace("&", " and ")))


def build(root: Path) -> dict:
    review_path = root / "step_3_moodboards" / "review" / "ellina_correction_manifest.json"
    csv_path = root / "step_2_text_embeddings" / "brand_metadata_today.csv"
    overlay_path = root / "ios" / "BrandMoodboardFinder" / "BrandMoodboardFinder" / "BrandMoodboards" / "v2_catalog_additions.json"
    output_path = root / "step_3_moodboards" / "review" / "reference_sources.json"

    records = json.loads(review_path.read_text(encoding="utf-8"))["records"]
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        catalog = list(csv.DictReader(stream))
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))["entries"]

    official_urls = {
        (normalized(row["brand_name"]), row["category"]): row["official_website"].strip()
        for row in catalog
        if row.get("official_website", "").strip()
    }
    official_urls.update({
        (normalized(row["brand_name"]), row["category"]): row["official_website"].strip()
        for row in overlay
        if row.get("official_website", "").strip()
    })

    sources = []
    for record in records:
        slug = normalized(record["brand_name"])
        key = (slug, record["category"])
        official_url = official_urls.get(key)
        if not official_url:
            raise RuntimeError(f"Missing official website for {record['review_id']}: {record['brand_name']} / {record['category']}")
        asset_name = f"{record['category']}__{slug}.jpg"
        query = f"{record['brand_name']} official {record['category']}"
        sources.append({
            "review_id": record["review_id"],
            "brand_name": record["brand_name"],
            "category": record["category"],
            "official_url": official_url,
            "google_images_url": f"https://www.google.com/search?udm=2&q={quote_plus(query)}",
            "screenshot_path": f"step_3_moodboards/review/reference_screenshots/{asset_name}",
        })

    payload = {
        "schema_version": 1,
        "source_csv": str(csv_path.relative_to(root)),
        "capture_policy": "Official URL from catalog; visual reference captured from one focused Google Images query per brand/category.",
        "records": sources,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Created {output_path} with {len(sources)} fixtures and {len({row['screenshot_path'] for row in sources})} unique captures.")
    return payload


def process_captures(root: Path, payload: dict) -> None:
    raw_dir = root / "tmp" / "pdfs" / "reference_raw"
    unique_paths = sorted({record["screenshot_path"] for record in payload["records"]})
    missing = []
    for relative_path in unique_paths:
        output = root / relative_path
        raw = raw_dir / f"{output.stem}.png"
        if not raw.exists():
            missing.append(str(raw))
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(raw) as opened:
            screenshot = opened.convert("RGB")
            portrait = ImageOps.fit(
                screenshot,
                (600, 900),
                method=Image.Resampling.LANCZOS,
                centering=(0.45, 0.5),
            )
            portrait.save(output, format="JPEG", quality=70, optimize=True, progressive=True)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} raw captures; first missing: {missing[0]}")
    print(f"Processed {len(unique_paths)} reference screenshots.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--process-captures",
        action="store_true",
        help="Crop and compress raw browser PNGs from tmp/pdfs/reference_raw after rebuilding the manifest.",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    manifest = build(project_root)
    if args.process_captures:
        process_captures(project_root, manifest)
