#!/usr/bin/env python3
"""Append the hand-checked Bags N-Z review while preserving stable review IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_ellina_manifest import (
    BAGS_N_Z_ADDITIONS,
    BAGS_N_Z_EXISTING,
    classify,
    issue_tags,
)


DEFAULT_MANIFEST = Path(__file__).with_name("ellina_correction_manifest.json")


def append_records(payload: dict) -> dict:
    records = payload["records"]
    if any(record.get("source_pdf") == "Bags N-Z.pdf" for record in records):
        return payload

    next_id = max(int(record["review_id"].split("-")[1]) for record in records) + 1
    for brand_name, note in BAGS_N_Z_EXISTING:
        action, priority, approval = classify(note)
        records.append({
            "review_id": f"ER-{next_id:03d}",
            "category": "bags",
            "brand_name": brand_name,
            "request_type": "existing_board",
            "recommended_action": action,
            "priority": priority,
            "approval_status": approval,
            "issue_tags": issue_tags(note),
            "reviewer_note": note,
            "source_pdf": "Bags N-Z.pdf",
            "source_page": 1,
            "implementation_status": (
                "needs_catalog_verification" if action == "catalog_verify" else "not_started"
            ),
            "asset_version": "v2",
        })
        next_id += 1

    for brand_name in BAGS_N_Z_ADDITIONS:
        records.append({
            "review_id": f"ER-{next_id:03d}",
            "category": "bags",
            "brand_name": brand_name,
            "request_type": "new_brand",
            "recommended_action": "catalog_add_and_generate",
            "priority": "P1",
            "approval_status": "needs_scope_and_catalog_verification",
            "issue_tags": ["requested_addition"],
            "reviewer_note": "Add this brand and create a category-appropriate bag moodboard.",
            "source_pdf": "Bags N-Z.pdf",
            "source_page": 1,
            "implementation_status": "not_started",
            "asset_version": "v2",
        })
        next_id += 1
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    before = len(payload["records"])
    append_records(payload)
    args.manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Manifest records: {before} -> {len(payload['records'])}")


if __name__ == "__main__":
    main()
