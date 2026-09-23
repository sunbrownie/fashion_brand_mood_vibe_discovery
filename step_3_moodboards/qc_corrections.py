#!/usr/bin/env python3
"""Run reproducible pre-review checks on corrected moodboard candidates."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "moodboards_creation" / "moodboards_corrected" / "v2"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def average_hash(image: Image.Image, size: int = 16) -> np.ndarray:
    gray = image.convert("L").resize((size, size))
    values = np.asarray(gray, dtype=np.float32)
    return values >= values.mean()


def duplicate_tile_screen(image: Image.Image) -> list[dict]:
    """Flag very similar non-adjacent tiles as a screening signal, not a final verdict."""
    width, height = image.size
    hashes = []
    for row in range(3):
        for col in range(2):
            left = round(col * width / 2)
            right = round((col + 1) * width / 2)
            top = round(row * height / 3)
            bottom = round((row + 1) * height / 3)
            hashes.append(((row, col), average_hash(image.crop((left, top, right, bottom)))))
    flags = []
    for index, (position, tile_hash) in enumerate(hashes):
        for other_position, other_hash in hashes[index + 1 :]:
            distance = int(np.count_nonzero(tile_hash != other_hash))
            if distance <= 8:
                flags.append({"tile_a": position, "tile_b": other_position, "hash_distance": distance})
    return flags


def ocr_screen(path: Path) -> dict:
    executable = shutil.which("tesseract")
    if executable is None:
        return {"status": "not_available", "text": ""}
    result = subprocess.run(
        [executable, str(path), "stdout", "--psm", "11"],
        check=False,
        capture_output=True,
        text=True,
    )
    text = " ".join(result.stdout.split())
    return {"status": "complete", "text": text}


def inspect(path: Path) -> dict:
    with Image.open(path) as image:
        width, height = image.size
        ratio = width / height
        duplicate_flags = duplicate_tile_screen(image)
    ocr = ocr_screen(path)
    failures = []
    if width < 900 or height < 1200:
        failures.append("resolution_below_minimum")
    if not 0.55 <= ratio <= 0.85:
        failures.append("unexpected_aspect_ratio")
    if ocr["status"] == "complete" and len(ocr["text"]) >= 3:
        failures.append("possible_embedded_text")
    if duplicate_flags:
        failures.append("possible_repeated_regions")
    return {
        "path": str(path),
        "width": width,
        "height": height,
        "aspect_ratio": round(ratio, 4),
        "ocr": ocr,
        "duplicate_screen": duplicate_flags,
        "automated_status": "pass" if not failures else "review",
        "failures": failures,
        "human_review_required": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_INPUT / "qc_report.json")
    args = parser.parse_args()
    paths = args.paths or sorted(
        path for path in args.input_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    report = {
        "note": "Automated duplicate detection is a coarse screen. Human brand-vibe and category review is mandatory.",
        "results": [inspect(path) for path in paths],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
