#!/usr/bin/env python3
"""Write approved Bags N-Z candidate and V2 paths back to the review manifest."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("ellina_correction_manifest.json")
REVIEWS = ROOT / "moodboards_creation" / "moodboards_corrected" / "v2" / "candidate_reviews.json"
V2_MANIFEST = ROOT / "ios" / "BrandMoodboardFinder" / "BrandMoodboardFinder" / "BrandMoodboards" / "v2_manifest.json"


def main() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    latest = {item["review_id"]: item for item in json.loads(REVIEWS.read_text(encoding="utf-8"))["reviews"]}
    v2_entries = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))["entries"]
    ios_paths = {
        review_id: entry["ios_v2_path"]
        for entry in v2_entries
        for review_id in entry.get("review_ids", [entry["review_id"]])
    }
    updated = 0
    for record in payload["records"]:
        if record.get("source_pdf") != "Bags N-Z.pdf":
            continue
        review = latest.get(record["review_id"])
        if record["recommended_action"] == "catalog_verify" and not review:
            record.update({
                "implementation_status": "needs_catalog_verification",
                "resolved_action": "pending_catalog_decision",
            })
            continue
        if not review or review.get("status") != "approved":
            continue
        record.update({
            "approval_status": "self_review_approved",
            "implementation_status": "implemented_v2",
            "resolved_action": (
                "catalog_added_and_generated"
                if record["recommended_action"] == "catalog_add_and_generate"
                else "regenerated"
            ),
            "approved_candidate_path": review["candidate_path"],
            "ios_v2_paths": [ios_paths[record["review_id"]]],
            "self_review_note": review["notes"],
        })
        updated += 1
    MANIFEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Finalized {updated} Bags N-Z records.")


if __name__ == "__main__":
    main()
