#!/usr/bin/env python3
"""Import one built-in ImageGen result into the resumable correction queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT / "moodboards_creation" / "moodboards_corrected" / "v2" / "generation_queue.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_id")
    parser.add_argument("generated_image", type=Path)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    args = parser.parse_args()

    queue = json.loads(args.queue.read_text(encoding="utf-8"))["queue"]
    item = next((entry for entry in queue if entry["review_id"] == args.review_id), None)
    if item is None:
        raise KeyError(f"Review ID not found in queue: {args.review_id}")
    source = args.generated_image.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    target = ROOT / item["candidate_path"]
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    sidecar = target.with_suffix(target.suffix + ".json")
    sidecar.write_text(json.dumps({
        "review_id": item["review_id"],
        "brand_name": item["brand_name"],
        "category": item["category"],
        "candidate": int(target.stem.rsplit("_", 1)[-1]),
        "model": "built-in image_gen",
        "prompt": item["prompt"],
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "status": "candidate_generated",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
