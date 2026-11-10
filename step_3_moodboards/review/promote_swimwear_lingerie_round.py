#!/usr/bin/env python3
"""Promote Ellina's swimwear/lingerie review assets into the current bundles."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]
TMP = ROOT / "tmp" / "swimwear_lingerie_updates"
CORRECTED = ROOT / "moodboards_creation" / "moodboards_corrected" / "v2" / "swimwear_lingerie"
CANONICAL = ROOT / "moodboards_creation" / "moodboards" / "swimwear_lingerie"
IOS = ROOT / "ios" / "BrandMoodboardFinder" / "BrandMoodboardFinder" / "BrandMoodboards"
HF = ROOT / "recommender" / "hosting_bundle" / "data" / "moodboards" / "swimwear_lingerie"
MANIFEST = IOS / "v2_manifest.json"
OVERLAY = IOS / "v2_catalog_additions.json"


SPECS = (
    ("R4-SWIM-ACACIA", "Acacia", "acacia", False),
    ("R4-SWIM-ARAKS", "Araks", "araks", False),
    ("R4-SWIM-BASERANGE", "Baserange", "baserange", False),
    ("R4-SWIM-FRUITY-BOOTY", "Fruity Booty", "fruity_booty", False),
    ("R4-SWIM-INTIMISSIMI", "Intimissimi", "intimissimi", False),
    ("R4-SWIM-LEFT-ON-FRIDAY", "Left On Friday", "left_on_friday", False),
    ("R4-SWIM-NATORI", "Natori", "natori", False),
    ("R4-SWIM-ONLY-HEARTS", "Only Hearts", "only_hearts", False),
    ("R4-SWIM-CALVIN-KLEIN", "Calvin Klein", "calvin_klein", True),
    ("R4-SWIM-LA-PERLA", "La Perla", "la_perla", True),
    ("R4-SWIM-CALZEDONIA", "Calzedonia", "calzedonia", True),
    ("R4-SWIM-BILLABONG", "Billabong", "billabong", True),
    ("R4-SWIM-ROXY", "Roxy", "roxy", True),
    ("R4-SWIM-WOLFORD", "Wolford", "wolford", True),
)

ADDITIONS = {
    "calvin_klein": ("https://www.calvinklein.co.uk", True, True, "mid price point", 1, "SKIMS"),
    "la_perla": ("https://www.laperla.com", False, True, "luxury", 1, "Natori"),
    "calzedonia": ("https://www.calzedonia.com", False, True, "affordable", 1, "Solid and Striped"),
    "billabong": ("https://www.billabong.com", True, True, "affordable", 1, "Bather"),
    "roxy": ("https://www.roxy-uk.co.uk", False, True, "affordable", 1, "Left On Friday"),
    "wolford": ("https://www.wolford.com", False, True, "upper price point", 1, "SKIMS"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_jpeg(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(
            image,
            (1024, 1536),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        image.save(target, "JPEG", quality=92, optimize=True, progressive=True)


def main() -> None:
    for directory in (CORRECTED, CANONICAL, IOS, HF):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    replace_keys = {("swimwear_lingerie", brand.casefold()) for _, brand, _, _ in SPECS}
    manifest["entries"] = [
        entry for entry in manifest["entries"]
        if (str(entry["category"]), str(entry["brand_name"]).casefold()) not in replace_keys
    ]

    for review_id, brand, slug, _is_new in SPECS:
        source = TMP / f"{slug}.png"
        if not source.is_file():
            raise FileNotFoundError(source)
        corrected = CORRECTED / f"{slug}.png"
        shutil.copy2(source, corrected)
        canonical = CANONICAL / f"{slug}.jpg"
        ios_asset = IOS / f"v2__swimwear_lingerie__{slug}.jpg"
        save_jpeg(source, canonical)
        save_jpeg(source, ios_asset)
        shutil.copy2(ios_asset, HF / f"{slug}.jpg")
        manifest["entries"].append({
            "review_id": review_id,
            "review_ids": [review_id],
            "category": "swimwear_lingerie",
            "brand_name": brand,
            "slug": slug,
            "source_path": f"moodboards_creation/moodboards_corrected/v2/swimwear_lingerie/{slug}.png",
            "analysis_v2_path": f"moodboards_creation/moodboards_corrected/v2/swimwear_lingerie/{slug}.png",
            "ios_v2_path": f"BrandMoodboardFinder/BrandMoodboards/v2__swimwear_lingerie__{slug}.jpg",
            "source_sha256": sha256(corrected),
            "ios_sha256": sha256(ios_asset),
            "rollback": "Restore the prior canonical asset from Git history or disable this v2__ asset.",
        })
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    overlay = json.loads(OVERLAY.read_text(encoding="utf-8"))
    new_names = {brand.casefold() for _, brand, _, is_new in SPECS if is_new}
    overlay["entries"] = [
        entry for entry in overlay["entries"]
        if not (
            str(entry["category"]) == "swimwear_lingerie"
            and str(entry["brand_name"]).casefold() in new_names
        )
    ]
    for review_id, brand, slug, is_new in SPECS:
        if not is_new:
            continue
        website, male, female, price, popularity, proxy = ADDITIONS[slug]
        overlay["entries"].append({
            "review_id": review_id,
            "brand_name": brand,
            "category": "swimwear_lingerie",
            "official_website": website,
            "male": male,
            "female": female,
            "price": price,
            "popularity": popularity,
            "moodboard_asset_name": f"v2__swimwear_lingerie__{slug}.jpg",
            "taste_proxy_brand_name": proxy,
        })
    OVERLAY.write_text(json.dumps(overlay, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for slug in ("perle_by_lola", "von_dutch"):
        for target in (HF / f"{slug}.jpg", IOS / f"v2__swimwear_lingerie__{slug}.jpg"):
            if target.exists():
                target.unlink()

    print(f"Promoted {len(SPECS)} boards; manifest={len(manifest['entries'])}, overlay={len(overlay['entries'])}")


if __name__ == "__main__":
    main()
