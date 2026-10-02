#!/usr/bin/env python3
"""Record append-only human review decisions for generated moodboard candidates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "moodboards_creation"
    / "moodboards_corrected"
    / "v2"
    / "candidate_reviews.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_id")
    parser.add_argument("candidate_path", type=Path)
    parser.add_argument("status", choices=["approved", "rejected", "pending"])
    parser.add_argument("notes")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output
    payload = (
        json.loads(output.read_text(encoding="utf-8"))
        if output.exists()
        else {"schema_version": 1, "reviews": []}
    )
    relative_path = str(args.candidate_path.resolve().relative_to(ROOT))
    decision = {
        "review_id": args.review_id,
        "candidate_path": relative_path,
        "status": args.status,
        "notes": args.notes,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["reviews"] = [
        item
        for item in payload["reviews"]
        if not (
            item["review_id"] == args.review_id
            and item["candidate_path"] == relative_path
        )
    ]
    payload["reviews"].append(decision)
    payload["reviews"].sort(key=lambda item: (item["review_id"], item["candidate_path"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {args.status}: {args.review_id} {relative_path}")


if __name__ == "__main__":
    main()
