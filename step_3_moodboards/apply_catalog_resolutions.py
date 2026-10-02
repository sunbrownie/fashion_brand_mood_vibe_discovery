#!/usr/bin/env python3
"""Record reviewed catalog resolutions and completed V2 moodboard assets."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "step_3_moodboards" / "review" / "ellina_correction_manifest.json"
REVIEWS_PATH = ROOT / "moodboards_creation" / "moodboards_corrected" / "v2" / "candidate_reviews.json"
V2_MANIFEST_PATH = (
    ROOT
    / "ios"
    / "BrandMoodboardFinder"
    / "BrandMoodboardFinder"
    / "BrandMoodboards"
    / "v2_manifest.json"
)
CATALOG_ADDITIONS_PATH = V2_MANIFEST_PATH.with_name("v2_catalog_additions.json")


VERIFIED_RESOLUTIONS = {
    "ER-008": {
        "resolved_action": "retire_category",
        "resolution_note": "Remove from clothes: Barc London is a pet-goods brand; its human item is a matching novelty jumper, not a viable apparel range.",
        "evidence_urls": ["https://www.barclondon.com/"],
    },
    "ER-046": {
        "resolved_action": "keep_and_regenerate",
        "resolution_note": "Keep in clothes: Herschel currently carries a substantial apparel range. Replace the duplicate-heavy board with the approved apparel-only V2.",
        "evidence_urls": ["https://herschel.eu/collections/apparel"],
    },
    "ER-175": {
        "resolved_action": "retire_category",
        "resolution_note": "Remove from shoes: the catalog URL resolves to a multi-brand children's retailer, and no distinct Childs footwear label could be verified.",
        "evidence_urls": ["https://childsplayclothing.com/"],
    },
    "ER-234": {
        "resolved_action": "retire_category",
        "resolution_note": "Remove from bags: the bags on the cited store are other labels such as INDISPENSABLE, not a sufficient Anonymous Ism bag range.",
        "evidence_urls": ["https://anonymousism.com/collections"],
    },
    "ER-244": {
        "resolved_action": "keep_and_regenerate",
        "resolution_note": "Keep in bags: Collina Strada has a current handbag category. Replace the off-vibe products with the approved brand-specific V2.",
        "evidence_urls": ["https://collinastrada.com/collections/bags"],
    },
    "ER-253": {
        "resolved_action": "retire_category",
        "resolution_note": "Remove from bags: Filling Pieces' current official range is footwear and apparel; a promotional tote is not a standalone bag range.",
        "evidence_urls": ["https://www.fillingpieces.com/"],
    },
    "ER-254": {
        "resolved_action": "retire_category",
        "resolution_note": "Remove from bags: Free Society's official label is a swimwear range and does not support a meaningful bag category.",
        "evidence_urls": ["https://www.freesocietyswimwear.com/pt/"],
    },
    "ER-264": {
        "resolved_action": "retire_category",
        "resolution_note": "Remove from bags: L'AGENCE's current official accessories navigation does not contain a handbag range.",
        "evidence_urls": ["https://lagence.com/"],
    },
    "ER-272": {
        "resolved_action": "retire_category",
        "resolution_note": "Remove from bags: MARK.S is a clothing label, while the catalog URL incorrectly points to Marks & Spencer; no MARK.S bag range was verified.",
        "evidence_urls": ["https://www.unanegozio.com/collections/mark-s"],
    },
}


def latest_reviews() -> dict[str, dict]:
    payload = json.loads(REVIEWS_PATH.read_text(encoding="utf-8"))
    result: dict[str, dict] = {}
    for review in payload["reviews"]:
        result[review["review_id"]] = review
    return result


def v2_assets_by_review_id() -> dict[str, list[str]]:
    payload = json.loads(V2_MANIFEST_PATH.read_text(encoding="utf-8"))
    result: dict[str, list[str]] = {}
    for entry in payload["entries"]:
        for review_id in entry.get("review_ids", [entry["review_id"]]):
            result.setdefault(review_id, []).append(entry["ios_v2_path"])
    return result


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    reviews = latest_reviews()
    assets = v2_assets_by_review_id()
    additions = {
        item["review_id"]: item
        for item in json.loads(CATALOG_ADDITIONS_PATH.read_text(encoding="utf-8"))["entries"]
    }
    completed = 0

    for record in manifest["records"]:
        review_id = record["review_id"]
        action = record["recommended_action"]
        if action in {"targeted_regenerate", "full_regenerate"}:
            review = reviews.get(review_id)
            if not review or review.get("status") != "approved":
                continue
            record["implementation_status"] = "implemented_v2"
            record["approval_status"] = "self_review_approved"
            record["resolved_action"] = "regenerated"
            record["approved_candidate_path"] = review["candidate_path"]
            record["ios_v2_paths"] = assets.get(review_id, [])
            record["self_review_note"] = review["notes"]
            completed += 1
        elif action == "catalog_add_and_generate":
            review = reviews.get(review_id)
            addition = additions.get(review_id)
            if not review or review.get("status") != "approved" or not addition or not assets.get(review_id):
                continue
            record["implementation_status"] = "implemented_v2"
            record["approval_status"] = "self_review_approved"
            record["resolved_action"] = "catalog_added_and_generated"
            record["approved_candidate_path"] = review["candidate_path"]
            record["ios_v2_paths"] = assets[review_id]
            record["catalog_overlay_path"] = str(CATALOG_ADDITIONS_PATH.relative_to(ROOT))
            record["taste_proxy_brand_name"] = addition["taste_proxy_brand_name"]
            record["self_review_note"] = review["notes"]
            completed += 1
        elif action == "catalog_retire_or_move":
            record["implementation_status"] = "implemented_catalog_v2"
            record["approval_status"] = "resolved"
            record["resolved_action"] = "retire_category"
            record["resolution_note"] = "Accepted Elena's category correction; the brand is filtered only from the reviewed category."
            completed += 1
        elif action == "catalog_merge":
            record["implementation_status"] = "implemented_catalog_v2"
            record["approval_status"] = "resolved"
            record["resolved_action"] = "merge_duplicate"
            record["resolution_note"] = "Merged the duplicate clothes entries by filtering Outland and retaining the canonical Outland Denim entry."
            completed += 1
        elif action == "catalog_verify":
            resolution = VERIFIED_RESOLUTIONS[review_id]
            record.update(resolution)
            record["verified_at"] = "2026-10-02"
            record["implementation_status"] = "implemented_v2" if resolution["resolved_action"] == "keep_and_regenerate" else "implemented_catalog_v2"
            record["approval_status"] = "self_review_approved" if resolution["resolved_action"] == "keep_and_regenerate" else "resolved"
            if resolution["resolved_action"] == "keep_and_regenerate":
                review = reviews[review_id]
                record["approved_candidate_path"] = review["candidate_path"]
                record["ios_v2_paths"] = assets.get(review_id, [])
                record["self_review_note"] = review["notes"]
            completed += 1

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Recorded {completed} completed fixtures in the Ellina manifest.")


if __name__ == "__main__":
    main()
