#!/usr/bin/env python3
"""Recover referenced Hugging Face moodboards that are still local LFS pointers."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "recommender" / "hosting_bundle"
DATA = HOST / "data"
REPO_ID = "Zilorina/brand-moodboard-recommender"
POINTER_HEADER = b"version https://git-lfs.github.com/spec/v1"


def main() -> None:
    metadata = pd.read_csv(DATA / "brand_metadata.csv")
    recovered = 0
    for relative in metadata["moodboard_path"].drop_duplicates():
        target = DATA / relative
        if not target.read_bytes()[:80].startswith(POINTER_HEADER):
            continue
        downloaded = Path(hf_hub_download(REPO_ID, filename=f"data/{relative}", repo_type="space"))
        shutil.copy2(downloaded, target)
        recovered += 1
        print(f"Recovered {relative}", flush=True)

    invalid = []
    for relative in metadata["moodboard_path"].drop_duplicates():
        target = DATA / relative
        try:
            with Image.open(target) as image:
                image.verify()
        except Exception as exc:
            invalid.append((relative, str(exc)))
    if invalid:
        raise RuntimeError(f"Invalid referenced moodboards after recovery: {invalid[:5]}")
    print(f"Recovered {recovered} LFS-backed moodboards; all referenced images are valid.")


if __name__ == "__main__":
    main()
