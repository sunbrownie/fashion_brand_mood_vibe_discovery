#!/usr/bin/env python3
"""Promote the November review additions into the latest-only iOS V2 bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
APP_ASSETS = ROOT / "ios" / "BrandMoodboardFinder" / "BrandMoodboardFinder" / "BrandMoodboards"
MANIFEST = APP_ASSETS / "v2_manifest.json"

UPDATES = [
    ("shoes", "Celine", "celine", "ER-169"),
    ("shoes", "Comme des Garçons", "comme_des_garcons", "ER-180"),
    ("shoes", "Comme des Garçons PLAY", "comme_des_garcons_play", "ER-180-PLAY"),
    ("bags", "CASABLANCA", "casablanca", "ER-238"),
    ("bags", "Chan Luu", "chan_luu", "ER-240"),
    ("bags", "Sandy Liang", "sandy_liang", "R3-BAG-SANDY-LIANG"),
    ("clothes", "Jacquemus", "jacquemus", "R3-GHOST-JACQUEMUS"),
    ("clothes", "Karoline Vitto", "karoline_vitto", "R3-GHOST-KAROLINE-VITTO"),
    ("clothes", "Montane", "montane", "R3-GHOST-MONTANE"),
    ("clothes", "Oner Active", "oner_active", "R3-GHOST-ONER-ACTIVE"),
    ("shoes", "Alohas", "alohas", "R3-SHOE-ALOHAS"),
    ("shoes", "Aldo", "aldo", "R3-SHOE-ALDO"),
    ("shoes", "Melissa", "melissa", "R3-SHOE-MELISSA"),
    ("shoes", "On Running", "on_running", "R3-SHOE-ON-RUNNING"),
    ("shoes", "Sam Edelman", "sam_edelman", "R3-SHOE-SAM-EDELMAN"),
    ("shoes", "Vibram", "vibram", "R3-SHOE-VIBRAM"),
    ("shoes", "Onitsuka Tiger", "onitsuka_tiger", "R3-SHOE-ONITSUKA-TIGER"),
    ("shoes", "Saucony", "saucony", "R3-SHOE-SAUCONY"),
    ("shoes", "Timberland", "timberland", "R3-SHOE-TIMBERLAND"),
    ("shoes", "Vans", "vans", "R3-SHOE-VANS"),
    ("shoes", "UGG", "ugg", "R3-SHOE-UGG"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image = ImageOps.fit(image, (1024, 1536), method=Image.Resampling.LANCZOS)
        image.save(target, "JPEG", quality=92, optimize=True, progressive=True)


def main() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = payload["entries"]
    lookup = {(entry["category"], entry["slug"]): entry for entry in entries}

    for category, brand_name, slug, review_id in UPDATES:
        source_rel = f"moodboards_creation/moodboards_corrected/v2/{category}/{slug}.png"
        source = ROOT / source_rel
        target_name = f"v2__{category}__{slug}.jpg"
        target = APP_ASSETS / target_name
        if not source.is_file():
            raise FileNotFoundError(source)
        export(source, target)
        key = (category, slug)
        entry = lookup.get(key)
        if entry is None:
            entry = {
                "review_id": review_id,
                "review_ids": [review_id],
                "category": category,
                "brand_name": brand_name,
                "slug": slug,
                "source_path": source_rel,
                "analysis_v2_path": source_rel,
                "ios_v2_path": f"BrandMoodboardFinder/BrandMoodboards/{target_name}",
                "rollback": "Restore the prior canonical asset from Git history or disable this v2__ asset.",
            }
            entries.append(entry)
            lookup[key] = entry
        else:
            review_ids = list(dict.fromkeys([*entry.get("review_ids", []), review_id]))
            entry.update({
                "review_id": review_id,
                "review_ids": review_ids,
                "brand_name": brand_name,
                "source_path": source_rel,
                "analysis_v2_path": source_rel,
                "ios_v2_path": f"BrandMoodboardFinder/BrandMoodboards/{target_name}",
            })
        entry["source_sha256"] = sha256(source)
        entry["ios_sha256"] = sha256(target)

    entries.sort(key=lambda entry: (entry["category"], entry["brand_name"].casefold(), entry["slug"]))
    payload["entries"] = entries
    MANIFEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Promoted {len(UPDATES)} assets; manifest now has {len(entries)} latest-only entries.")


if __name__ == "__main__":
    main()
