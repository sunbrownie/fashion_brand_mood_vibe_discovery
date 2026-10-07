#!/usr/bin/env python3
"""Build separate CLIP image embeddings for every hosted moodboard."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "recommender" / "hosting_bundle" / "data"
MODEL_NAME = "openai/clip-vit-base-patch32"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    metadata = pd.read_csv(DATA / "brand_metadata.csv")
    paths = [DATA / value for value in metadata["moodboard_path"]]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} moodboards; first missing: {missing[0]}")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
    model = CLIPModel.from_pretrained(MODEL_NAME, local_files_only=True)
    model.eval()

    batches = []
    with torch.inference_mode():
        for start in range(0, len(paths), args.batch_size):
            batch_paths = paths[start : start + args.batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as opened:
                    images.append(ImageOps.exif_transpose(opened).convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            features = model.get_image_features(**inputs)
            # transformers 5.x returns the vision-model output here rather than
            # the tensor returned by 4.x; its pooler_output is already the
            # projected 512-D CLIP image feature.
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-12)
            batches.append(features.cpu().numpy().astype("float32"))
            completed = min(start + args.batch_size, len(paths))
            if completed % 160 == 0 or completed == len(paths):
                print(f"Embedded {completed}/{len(paths)} moodboards.", flush=True)

    embeddings = np.concatenate(batches, axis=0)
    np.savez_compressed(
        DATA / "moodboard_image_embeddings_clip_vit_b32.npz",
        embeddings=embeddings,
        model=np.asarray(MODEL_NAME),
    )
    output_metadata = metadata[["brand_name", "category", "moodboard_path"]].copy()
    output_metadata.insert(0, "row_index", np.arange(len(output_metadata)))
    output_metadata["model"] = MODEL_NAME
    output_metadata["source_sha256"] = [sha256(path) for path in paths]
    output_metadata.to_csv(DATA / "moodboard_image_embeddings_metadata.csv", index=False)
    print(f"Wrote {embeddings.shape} CLIP moodboard embeddings for {len(metadata)} catalog rows.")


if __name__ == "__main__":
    main()
