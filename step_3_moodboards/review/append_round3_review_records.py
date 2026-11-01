#!/usr/bin/env python3
"""Append the latest Sandy Liang, shoe-addition, and presentation review pages."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "step_3_moodboards" / "review" / "round2_re_review_manifest.json"


def record(
    review_id: str,
    brand_name: str,
    category: str,
    slug: str,
    feedback: str,
    before_path: str,
    prompt: str,
    *,
    official_url: str,
    addition: bool = False,
) -> dict:
    return {
        "review_id": review_id,
        "source_page": None,
        "brand_name": brand_name,
        "category": category,
        "variant": None,
        "round2_feedback": feedback,
        "review_status": "latest review request",
        "round1_path": before_path,
        "updated_path": f"moodboards_creation/moodboards_corrected/v2/{category}/{slug}.png",
        "reference_path": f"step_3_moodboards/review/reference_screenshots/{category}__{slug}.jpg",
        "official_url": official_url,
        "generation_prompt": prompt,
        "comment_label": "LATEST REVIEW COMMENT",
        "panel_labels": [
            "PREVIOUS V2 STATUS" if addition else "PREVIOUS CURRENT BOARD",
            "LATEST UPDATED BOARD",
            "GOOGLE IMAGES REFERENCE",
        ],
    }


PLACEHOLDER = "step_3_moodboards/review/placeholders/not_in_v2_yet.png"

RECORDS = [
    record(
        "R3-BAG-SANDY-LIANG", "Sandy Liang", "bags", "sandy_liang",
        "Remove the elephant bag; it is a different brand.",
        "tmp/pdfs/round3_inputs/previous/bags/sandy_liang.png",
        "Sandy Liang handbag moodboard with distinct bowler, satin, baguette, mini top-handle, nylon tote and evening pouch silhouettes; no elephant or animal-shaped bag.",
        official_url="https://sandyliang.info/collections/bags",
    ),
    record(
        "R3-GHOST-JACQUEMUS", "Jacquemus", "clothes", "jacquemus",
        "Remove the invisible-person / ghost-mannequin effect; use visible models or true flat lays.",
        "tmp/pdfs/round3_inputs/previous/clothes/jacquemus.png",
        "Mediterranean Jacquemus clothing shown on clearly visible adult models or as genuine flat lays; no hollow or floating body-shaped garments.",
        official_url="https://www.jacquemus.com",
    ),
    record(
        "R3-GHOST-KAROLINE-VITTO", "Karoline Vitto", "clothes", "karoline_vitto",
        "Remove the invisible-person / ghost-mannequin effect; show the sculptural clothes on visible models.",
        "tmp/pdfs/round3_inputs/previous/clothes/karoline_vitto.png",
        "Body-positive Karoline Vitto dresses worn by clearly visible adult models of varied body sizes, with one metal-frame detail; no ghost mannequin.",
        official_url="https://karolinevitto.com",
    ),
    record(
        "R3-GHOST-MONTANE", "Montane", "clothes", "montane",
        "Remove the invisible-person / ghost-mannequin effect; use visible hikers, hangers or true flat lays.",
        "tmp/pdfs/round3_inputs/previous/clothes/montane.png",
        "Technical Montane shells, insulation and trousers worn by visible adult hikers or presented on visible hangers and true flat lays; no floating jacket forms.",
        official_url="https://montane.com",
    ),
    record(
        "R3-GHOST-ONER-ACTIVE", "Oner Active", "clothes", "oner_active",
        "Remove the invisible-person / ghost-mannequin effect; use visible models and folded or true flat lays.",
        "tmp/pdfs/round3_inputs/previous/clothes/oner_active.png",
        "Inclusive Oner Active looks on clearly visible adult women plus folded and true flat-lay garments; no floating body-shaped activewear.",
        official_url="https://uk.oneractive.com",
    ),
    record(
        "R3-SHOE-ALOHAS", "Alohas", "shoes", "alohas",
        "Add this missing shoe brand to the current V2 moodboard set.", PLACEHOLDER,
        "Alohas moodboard with ballet flat, loafer, crochet and suede retro sneakers, ankle boot and minimal sandal.",
        official_url="https://alohas.com/collections/shoes", addition=True,
    ),
    record(
        "R3-SHOE-ALDO", "Aldo", "shoes", "aldo",
        "Add this missing shoe moodboard to the current V2 set.", PLACEHOLDER,
        "Aldo moodboard with slingback, pump, metallic flat, loafer, clean sneaker, ankle boot and red heel.",
        official_url="https://www.aldoshoes.com", addition=True,
    ),
    record(
        "R3-SHOE-MELISSA", "Melissa", "shoes", "melissa",
        "Add this missing shoe brand to the current V2 moodboard set.", PLACEHOLDER,
        "Melissa jelly-shoe moodboard with translucent fisherman sandal, Mary Jane, platform, ballet flat, clear mule and rain boot.",
        official_url="https://www.melissa.com.br", addition=True,
    ),
    record(
        "R3-SHOE-ON-RUNNING", "On Running", "shoes", "on_running",
        "Add this missing shoe brand to the current V2 moodboard set.", PLACEHOLDER,
        "On moodboard with lifestyle, waterproof, cushioned road, race-day, trail and tennis-inspired CloudTec footwear.",
        official_url="https://www.on.com/en-gb/shop/shoes", addition=True,
    ),
    record(
        "R3-SHOE-SAM-EDELMAN", "Sam Edelman", "shoes", "sam_edelman",
        "Add this missing shoe brand to the current V2 moodboard set.", PLACEHOLDER,
        "Sam Edelman moodboard with slingback, evening sandal, loafer, western boot, Mary Jane, ballet flat and sneaker.",
        official_url="https://www.samedelman.com", addition=True,
    ),
    record(
        "R3-SHOE-VIBRAM", "Vibram", "shoes", "vibram",
        "Add this missing shoe brand to the current V2 moodboard set.", PLACEHOLDER,
        "Vibram moodboard focused on varied FiveFingers, barefoot trainers, water shoes, approach shoes and technical sole details.",
        official_url="https://www.vibram.com", addition=True,
    ),
    record(
        "R3-SHOE-ONITSUKA-TIGER", "Onitsuka Tiger", "shoes", "onitsuka_tiger",
        "Add this missing shoe brand to the current V2 moodboard set.", PLACEHOLDER,
        "Onitsuka Tiger moodboard spanning retro low-profile runners, court shoes, tennis shoes, futuristic trainers and high tops.",
        official_url="https://www.onitsukatiger.com/gb/en-gb/shoes/", addition=True,
    ),
    record(
        "R3-SHOE-SAUCONY", "Saucony", "shoes", "saucony",
        "Add this missing shoe brand to the current V2 moodboard set.", PLACEHOLDER,
        "Saucony moodboard with heritage runner, retro lifestyle sneaker, racing, cushioned road, trail and performance shoes.",
        official_url="https://www.saucony.com", addition=True,
    ),
    record(
        "R3-SHOE-TIMBERLAND", "Timberland", "shoes", "timberland",
        "Add this missing shoe brand to the current V2 moodboard set.", PLACEHOLDER,
        "Timberland moodboard with six-inch boot, field boot, hiking boot, trail sneaker, boat shoe and winter lug boot.",
        official_url="https://www.timberland.com", addition=True,
    ),
    record(
        "R3-SHOE-VANS", "Vans", "shoes", "vans",
        "Add this missing shoe brand to the current V2 moodboard set.", PLACEHOLDER,
        "Vans moodboard with low-top skate icon, checkerboard slip-on, canvas deck shoe, high top, chunky suede and trail styles.",
        official_url="https://www.vans.com/en-gb/c/shoes-00081", addition=True,
    ),
    record(
        "R3-SHOE-UGG", "UGG", "shoes", "ugg",
        "Add this missing shoe brand to the current V2 moodboard set.", PLACEHOLDER,
        "UGG moodboard with Ultra Mini, Tasman-style clog, platform mini, plush slipper, relaxed sneaker and tall suede boot.",
        official_url="https://www.ugg.com", addition=True,
    ),
]


def main() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    additions = {item["review_id"]: item for item in RECORDS}
    payload["records"] = [
        item for item in payload["records"] if item["review_id"] not in additions
    ] + RECORDS
    payload["schema_version"] = "round2-plus-latest-review-v1"
    MANIFEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Review manifest now contains {len(payload['records'])} pages.")


if __name__ == "__main__":
    main()
