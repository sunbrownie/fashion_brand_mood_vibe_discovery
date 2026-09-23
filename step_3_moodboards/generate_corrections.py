#!/usr/bin/env python3
"""Generate versioned moodboard correction candidates from Ellina's manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "step_3_moodboards" / "helpers"
if str(HELPERS) not in sys.path:
    sys.path.insert(0, str(HELPERS))

from moodboard_helpers import (  # noqa: E402
    brands_df,
    gemini_moodboard,
    moodboard_prompt,
    save_corrected_moodboard,
)


DEFAULT_MANIFEST = ROOT / "step_3_moodboards" / "review" / "ellina_correction_manifest.json"
IMAGE_ACTIONS = {"full_regenerate", "targeted_regenerate"}


def find_brand_row(category: str, brand_name: str) -> pd.Series:
    matches = brands_df[
        brands_df["category"].eq(category)
        & brands_df["brand_name"].str.casefold().eq(brand_name.casefold())
    ]
    if matches.empty:
        raise KeyError(f"No dataset row for {brand_name!r} in {category!r}.")
    return matches.iloc[0]


def select_records(payload: dict, args: argparse.Namespace) -> list[dict]:
    records = [
        record
        for record in payload["records"]
        if record["recommended_action"] in IMAGE_ACTIONS
        and record["approval_status"] == "approved_for_pilot"
    ]
    if args.category:
        records = [record for record in records if record["category"] == args.category]
    if args.brand:
        needle = args.brand.casefold()
        records = [record for record in records if record["brand_name"].casefold() == needle]
    if args.action:
        records = [record for record in records if record["recommended_action"] == args.action]
    return records[: args.limit] if args.limit else records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--category", choices=["clothes", "shoes", "bags", "jewellery"])
    parser.add_argument("--brand")
    parser.add_argument("--action", choices=sorted(IMAGE_ACTIONS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--generate", action="store_true", help="Call Gemini. Without this flag, print prompts only.")
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = select_records(payload, args)
    if not records:
        raise SystemExit("No correction records matched the filters.")

    for record in records:
        row = find_brand_row(record["category"], record["brand_name"])
        prompt = moodboard_prompt(row, reviewer_note=record["reviewer_note"])
        print(f"\n[{record['review_id']}] {record['category']} / {record['brand_name']}")
        print(prompt)
        if not args.generate:
            continue
        variants = args.variants if record["recommended_action"] == "full_regenerate" else 1
        for candidate in range(1, variants + 1):
            raw_bytes, mime_type = gemini_moodboard(prompt)
            path = save_corrected_moodboard(
                raw_bytes,
                mime_type,
                record["category"],
                record["brand_name"],
                candidate=candidate,
                prompt=prompt,
                review_id=record["review_id"],
            )
            print(f"saved {path}")


if __name__ == "__main__":
    main()
