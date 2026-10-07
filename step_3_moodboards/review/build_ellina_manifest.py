#!/usr/bin/env python3
"""Build the structured Ellina moodboard-review manifest from the review PDFs."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd


REVIEW_FILES = {
    "Clothes A-D.pdf": "clothes",
    "Clothes E-L.pdf": "clothes",
    "Clothes M-P.pdf": "clothes",
    "Clothes Q-Z.pdf": "clothes",
    "Shoes A-J-2.pdf": "shoes",
    "Jewelry R-z.pdf": "jewellery",
    # bags A-C is a duplicate subset of bags A-M.
    "bags A-M.pdf": "bags",
}

ADDITIONS = {
    "clothes": [
        "A Bathing Ape", "Abercrombie & Fitch", "Adidas", "Arket", "AVAVAV",
        "Barbour", "Balmain", "Bandit Running", "Billabong", "Bimba Y Lola",
        "Burton", "Burberry", "Calvin Klein / Calvin Klein Jeans", "Canada Goose",
        "Carhartt WIP", "Charles Jeffrey Loverboy", "Comme des Garcons Play",
        "Conner Ives", "Diesel", "Desigual", "Duran Lantink", "Erdem", "GAP",
        "Gant", "Good American", "Gymshark", "Gucci", "Hollister", "Jaded London",
        "Jean Paul Gaultier", "Karl Lagerfeld", "Lacoste", "Levi's", "Loro Piana",
        "Lululemon", "Mango", "Martine Rose", "Missoni", "Michael Kors", "Missguided",
        "McQueen", "MKI MIYUKI ZOKU", "Moncler", "Moschino", "Mugler", "Napapijri",
        "Needles", "Never Fully Dressed", "Neighbourhood", "New Balance", "Nike",
        "No Problemo", "Oner Active", "Open YY", "Palm Angels", "Paul Smith",
        "Polo Ralph Lauren", "Puma", "P.E Nation", "Pull & Bear",
    ],
    "shoes": [
        "ABRA", "Air Jordan", "Asics", "Adidas", "Golden Goose", "Birkenstock",
        "EYTYS", "Geox", "Gia Borghini", "Church's", "Gucci", "Crocs", "Clarks",
        "Ecco", "Converse", "Carvela", "Diemme", "Hunter",
    ],
    "jewellery": [
        "Vivienne Westwood", "Roxanne Assoulin", "Roxanne First", "Sonia Petroff",
        "Zoe Mohm", "William Welstead", "YVMIN", "Yellow Swallow",
    ],
}

# Category coverage additions requested after the original 360-fixture review.
# These are appended after the original records so ER-001 through ER-360 remain stable.
CATEGORY_EXPANSIONS = [
    ("shoes", "New Balance"),
    ("shoes", "Nike"),
    ("shoes", "Puma"),
    ("shoes", "Burton"),
]

ADDITION_FEEDBACK_OVERRIDES = {
    ("clothes", "Bimba Y Lola"): "The moodboard feels off-vibe compared with current Google and official imagery; correct the styling and palette.",
    ("clothes", "Burton"): "The moodboard feels off compared with current Google and official imagery; make the technical snowboard range more representative.",
    ("clothes", "Conner Ives"): "The moodboard feels off-vibe compared with current imagery; show a broader, more eclectic and crafted range.",
    ("clothes", "Diesel"): "The moodboard is too denim-only; include the broader current clothing range and correct the vibe.",
    ("clothes", "Desigual"): "The moodboard feels off-vibe; make it closer to the current modern brand styling rather than an older generic bohemian direction.",
    ("clothes", "McQueen"): "The moodboard feels off-vibe; represent the current sculptural range beyond only black gothic tailoring.",
    ("clothes", "Never Fully Dressed"): "The moodboard feels off-vibe compared with current imagery; make it more elevated, versatile and representative.",
    ("clothes", "Neighbourhood"): "The moodboard may not be fully on-vibe; correct it using current Google and official imagery.",
    ("clothes", "No Problemo"): "Include core shirts and sweatshirts carrying the NO PROBLEMO logo as well as the wider streetwear range.",
    ("shoes", "EYTYS"): "The board repeats boots; show a broader set of distinct current footwear forms.",
    ("shoes", "Geox"): "The moodboard feels off-vibe; correct it against current Google and official imagery.",
    ("jewellery", "Yellow Swallow"): "The moodboard feels off-vibe compared with current imagery; replace the dark direction with the brand's brighter jewelry language.",
}

HEADER_OVERRIDES = {
    "garcons": ("Comme des Garcons", 2),
    "q": ("Cydwoq", 2),
    "sander": ("Jil Sander", 2),
    "e": ("Juicy Couture", 2),
    "u": ("Thebe Magugu", 2),
}

GLOBAL_RULES = [
    "Use no more than two images of the same person on one board.",
    "Do not repeat a product, detail crop, colourway, model photo, or setting.",
    "Vary collage devices to suit the brand. Do not default every board to bean shapes.",
    "Use interiors, streets, landscapes, and props only when they support the brand.",
    "Remove accidental words and labels unless a distinctive brand mark is explicitly approved.",
]

PILOT_STATUS = {
    "ER-001": "implemented_v2",
    "ER-173": "implemented_v2",
    "ER-221": "implemented_v2",
    "ER-236": "implemented_v2",
}


def squash(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def clean_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" /\n\t")
    value = re.sub(r"\s+([,?.])", r"\1", value)
    return value


def load_catalog(root: Path) -> dict[str, list[str]]:
    candidates = [
        root / "recommender" / "hosting_bundle" / "data" / "brand_metadata.csv",
        root / "step_2_text_embeddings" / "brand_metadata.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        result: dict[str, list[str]] = defaultdict(list)
        for _, row in frame.iterrows():
            category = str(row.get("category", "")).strip().lower()
            brand = str(row.get("brand_name", "")).strip()
            if category and brand and brand not in result[category]:
                result[category].append(brand)
        return dict(result)
    raise FileNotFoundError("No brand metadata CSV was found for name normalisation.")


def extract_blocks(pdf_path: Path, xml_path: Path) -> list[tuple[float, float, str]]:
    subprocess.run(
        ["pdftotext", "-bbox-layout", str(pdf_path), str(xml_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    ns = {"x": "http://www.w3.org/1999/xhtml"}
    root = ET.parse(xml_path).getroot()
    blocks = []
    for block in root.findall(".//x:block", ns):
        lines = []
        for line in block.findall("x:line", ns):
            words = [clean_text(word.text or "") for word in line.findall("x:word", ns)]
            lines.append(" ".join(word for word in words if word))
        text = clean_text(" ".join(lines))
        if text:
            blocks.append((float(block.attrib["xMin"]), float(block.attrib["yMin"]), text))
    return blocks


def split_grid(blocks: list[tuple[float, float, str]]) -> list[list[str]]:
    y_values = sorted({y for _, y, _ in blocks})
    bands: list[tuple[float, float]] = []
    start = previous = y_values[0]
    for y_value in y_values[1:]:
        if y_value - previous > 700:
            bands.append((start, previous))
            start = y_value
        previous = y_value
    bands.append((start, previous))

    cells: list[list[str]] = []
    for y_min, y_max in bands:
        row = [item for item in blocks if y_min - 1 <= item[1] <= y_max + 1]
        x_values = sorted({x for x, _, _ in row})
        if not x_values:
            continue
        x_groups: list[list[float]] = []
        current = [x_values[0]]
        for x_value in x_values[1:]:
            if x_value - current[-1] > 420:
                x_groups.append(current)
                current = [x_value]
            else:
                current.append(x_value)
        x_groups.append(current)
        for x_group in x_groups:
            items = sorted(
                [item for item in row if min(x_group) - 1 <= item[0] <= max(x_group) + 1],
                key=lambda item: (item[1], item[0]),
            )
            texts = [clean_text(item[2]) for item in items if clean_text(item[2])]
            if texts:
                cells.append(texts)
    return cells


def canonical_brand(raw_header: str, category: str, catalog: dict[str, list[str]]) -> tuple[str, int]:
    key = squash(raw_header)
    if key in HEADER_OVERRIDES:
        return HEADER_OVERRIDES[key]
    names = catalog.get(category, [])
    if not names:
        return clean_text(raw_header), 1
    match = max(names, key=lambda name: difflib.SequenceMatcher(None, key, squash(name)).ratio())
    score = difflib.SequenceMatcher(None, key, squash(match)).ratio()
    return (match if score >= 0.62 else clean_text(raw_header), 1)


def split_brand_and_note(texts: list[str], category: str, catalog: dict[str, list[str]]) -> tuple[str, str]:
    """Split a cell even when PDF extraction put its heading and note in one block."""
    joined = clean_text(" ".join(texts))
    compact = squash(joined)
    special_prefixes = {
        "garconscommedes": "Comme des Garçons",
        "qcydwo": "Cydwoq",
        "sanderjil": "Jil Sander",
        "ejuicycouture": "Juicy Couture",
        "uthebemagug": "Thebe Magugu",
        "ilippak": "Filippa K",
        "illingpieces": "Filling Pieces",
        "reesociety": "Free Society",
    }
    for prefix, brand in special_prefixes.items():
        if compact.startswith(prefix):
            target = prefix
            break
    else:
        matches = [name for name in catalog.get(category, []) if compact.startswith(squash(name))]
        if matches:
            brand = max(matches, key=lambda name: len(squash(name)))
            target = squash(brand)
        else:
            brand, consumed = canonical_brand(texts[0], category, catalog)
            return brand, clean_text(" ".join(texts[consumed:]))

    consumed_chars = 0
    matched = ""
    for index, char in enumerate(joined):
        piece = squash(char)
        if piece:
            matched += piece
        if matched == target:
            consumed_chars = index + 1
            break
    return brand, clean_text(joined[consumed_chars:])


def issue_tags(note: str) -> list[str]:
    text = note.lower()
    tags = []
    rules = {
        "duplicate_assets": ("repeat", "same image", "same item", "same person", "same woman", "same print"),
        "embedded_text": ("word", "text", "label", "ecovero"),
        "wrong_category": ("remove", "only bags", "only shoes", "only jewelry", "only jewellery", "don’t make", "dont make"),
        "off_brand": ("off vibe", "wrong vibe", "off brand", "different vibe", "bad collage", "none of the images"),
        "wrong_context": ("interior", "building", "carpet", "setting", "chairs"),
        "missing_signature": ("signature", "iconic", "popular", "classic", "logo", "2.55", "heart-shaped"),
        "casting": ("no women", "male-centric", "female images", "different people"),
        "insufficient_range": ("more colours", "more color", "variation", "more outfits", "more clothes", "more shoes", "more bags"),
        "unverified_brand_or_category": ("worth keeping", "could not verify", "can’t find", "cant find", "hard to find", "barely any"),
    }
    for tag, needles in rules.items():
        if any(needle in text for needle in needles):
            tags.append(tag)
    return tags or ["art_direction"]


def classify(note: str) -> tuple[str, str, str]:
    text = note.lower()
    compact = squash(note)
    if "mergeinto" in compact:
        return "catalog_merge", "P0", "needs_catalog_verification"
    catalog_verify_signals = (
        "worthkeeping",
        "worthremoving",
        "couldnotverify",
        "cantfind",
        "hardtofind",
        "barelyanybags",
        "rarelyhavebags",
        "notenoughbagstolookat",
        "mainlyfordogs",
    )
    if any(term in compact for term in catalog_verify_signals):
        return "catalog_verify", "P0", "needs_catalog_verification"
    catalog_retire_signals = (
        "doesntdoclothes",
        "noclothesremove",
        "onlybagsremove",
        "onlyjewelrynoclothes",
        "onlyjewellerynoclothes",
        "onlyjewerlynoclothes",
        "removefromclothes",
        "removefromhereandkeepforshoes",
        "removefromshoes",
        "barelydoanyshoes",
        "onlyjewelryandclothes",
        "onlyjewelleryandclothes",
        "rarelydoesshoes",
        "dontdoshoes",
        "dontmakeshoes",
        "noshoes",
        "dontmakejewelry",
        "dontmakejewellery",
        "mostlyresellnotmake",
        "mostlykeychains",
        "clothingbranddoesntdojewelry",
        "clothingbranddoesntdojewellery",
        "closetonojewelry",
        "closetonojewellery",
    )
    removal_justifications = (
        "nojewelry",
        "nojewellery",
        "onlyjewelry",
        "onlyjewellery",
        "onlyjewerly",
        "limitedjewelry",
        "limitedjewellery",
        "limitedtonojewelry",
        "limitedtonojewellery",
        "mostlyresell",
        "mostlykeychains",
    )
    explicit_removal = compact == "remove" or (
        compact.startswith("remove")
        and any(term in compact for term in removal_justifications)
    )
    if explicit_removal or any(term in compact for term in catalog_retire_signals):
        return "catalog_retire_or_move", "P0", "needs_catalog_verification"
    full_redo = (
        compact.startswith("offvibe")
        or compact.startswith("wrongvibe")
        or compact.startswith("differentvibe")
        or "completelyoff" in compact
        or "allimagesoff" in compact
        or "noneoftheimages" in compact
        or "redofully" in compact
        or "wrongbrandshoes" in compact
        or "collageisfromtheshoesection" in compact
        or "badcollage" in compact
    )
    if full_redo:
        return "full_regenerate", "P1", "approved_for_pilot"
    return "targeted_regenerate", "P2", "approved_for_pilot"


def build_manifest(root: Path, review_dir: Path) -> dict:
    catalog = load_catalog(root)
    records = []
    record_id = 1
    with tempfile.TemporaryDirectory(prefix="ellina-review-") as temp_dir:
        temp = Path(temp_dir)
        for source_file, category in REVIEW_FILES.items():
            pdf_path = review_dir / source_file
            if not pdf_path.exists():
                raise FileNotFoundError(pdf_path)
            blocks = extract_blocks(pdf_path, temp / f"{pdf_path.stem}.html")
            for texts in split_grid(blocks):
                joined = clean_text(" ".join(texts))
                if not joined or joined == "/":
                    continue
                first_key = squash(texts[0])
                joined_key = squash(joined)
                if (
                    first_key.startswith("overallnotes")
                    or first_key.startswith("toadd")
                    or first_key.startswith("note")
                    or joined_key.startswith("enotgetai")
                ):
                    continue
                brand, note = split_brand_and_note(texts, category, catalog)
                if brand in {"Filippa K", "Filling Pieces", "Free Society"} and note.endswith(" F"):
                    note = note[:-2]
                if not note:
                    continue
                action, priority, approval = classify(note)
                records.append({
                    "review_id": f"ER-{record_id:03d}",
                    "category": category,
                    "brand_name": brand,
                    "request_type": "existing_board",
                    "recommended_action": action,
                    "priority": priority,
                    "approval_status": approval,
                    "issue_tags": issue_tags(note),
                    "reviewer_note": note,
                    "source_pdf": source_file,
                    "source_page": 1,
                    "implementation_status": PILOT_STATUS.get(f"ER-{record_id:03d}", "not_started"),
                    "asset_version": "v2",
                })
                record_id += 1

    for category, brands in ADDITIONS.items():
        for brand in brands:
            records.append({
                "review_id": f"ER-{record_id:03d}",
                "category": category,
                "brand_name": brand,
                "request_type": "new_brand",
                "recommended_action": "catalog_add_and_generate",
                "priority": "P1",
                "approval_status": "needs_scope_and_catalog_verification",
                "issue_tags": ["requested_addition"],
                "reviewer_note": ADDITION_FEEDBACK_OVERRIDES.get(
                    (category, brand),
                    "Add this brand and create a category-appropriate moodboard.",
                ),
                "source_pdf": "Ellina add lists",
                "source_page": 1,
                "implementation_status": "not_started",
                "asset_version": "v2",
            })
            record_id += 1

    for category, brand in CATEGORY_EXPANSIONS:
        records.append({
            "review_id": f"ER-{record_id:03d}",
            "category": category,
            "brand_name": brand,
            "request_type": "new_brand_category",
            "recommended_action": "catalog_add_and_generate",
            "priority": "P1",
            "approval_status": "needs_scope_and_catalog_verification",
            "issue_tags": ["requested_category_expansion"],
            "reviewer_note": (
                "Add a separate shoe moodboard because footwear is a core, visually distinct "
                "part of this brand's current assortment."
            ),
            "source_pdf": "Category coverage audit",
            "source_page": 1,
            "implementation_status": "not_started",
            "asset_version": "v2",
        })
        record_id += 1

    return {
        "schema_version": 1,
        "generated_on": date.today().isoformat(),
        "reviewer": "Ellina",
        "source_directory": str(review_dir),
        "policy": {
            "original_assets_immutable": True,
            "corrected_asset_root": "moodboards_creation/moodboards_corrected/v2",
            "category_changes_require_verification": True,
            "new_brands_require_scope_verification": True,
            "global_rules": GLOBAL_RULES,
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("ellina_correction_manifest.json"))
    args = parser.parse_args()
    payload = build_manifest(args.root.resolve(), args.review_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(payload['records'])} records to {args.output}")


if __name__ == "__main__":
    main()
