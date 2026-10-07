#!/usr/bin/env python3
"""Audit V2 moodboard images against their production text-card descriptions."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "recommender" / "hosting_bundle" / "data"
V2_MANIFEST = ROOT / "ios" / "BrandMoodboardFinder" / "BrandMoodboardFinder" / "BrandMoodboards" / "v2_manifest.json"
APP_ASSETS = V2_MANIFEST.parent
REVIEW_MANIFEST = ROOT / "step_3_moodboards" / "review" / "ellina_correction_manifest.json"
OUTPUT = ROOT / "step_3_moodboards" / "review" / "v2_description_alignment_audit.csv"
MODEL_NAME = "openai/clip-vit-base-patch32"


def normalised(value: str) -> str:
    folded = "".join(
        character
        for character in unicodedata.normalize("NFKD", str(value).replace("&", " and "))
        if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[a-z0-9]+", folded.lower()))


def text_card(row: pd.Series) -> str:
    return " ".join(
        (
            f"Aesthetic: {row.get('aesthetic_keywords', '')}.",
            f"Silhouettes: {row.get('silhouettes', '')}.",
            f"Materials: {row.get('materials', '')}.",
            f"Palette: {row.get('palette', '')}.",
        )
    )


def encode_texts(model: CLIPModel, processor: CLIPProcessor, texts: list[str]) -> np.ndarray:
    output: list[np.ndarray] = []
    for start in range(0, len(texts), 64):
        batch = processor(text=texts[start : start + 64], padding=True, truncation=True, return_tensors="pt")
        with torch.inference_mode():
            values = model.get_text_features(**batch)
            if not isinstance(values, torch.Tensor):
                values = values.pooler_output
            values = torch.nn.functional.normalize(values, dim=-1)
        output.append(values.cpu().numpy())
    return np.concatenate(output).astype("float32")


def encode_images(model: CLIPModel, processor: CLIPProcessor, paths: list[Path]) -> np.ndarray:
    output: list[np.ndarray] = []
    for start in range(0, len(paths), 16):
        images = []
        for path in paths[start : start + 16]:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        batch = processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            values = model.get_image_features(**batch)
            if not isinstance(values, torch.Tensor):
                values = values.pooler_output
            values = torch.nn.functional.normalize(values, dim=-1)
        output.append(values.cpu().numpy())
    return np.concatenate(output).astype("float32")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    previous_lookup: dict[str, dict] = {}
    if args.output.exists():
        previous = pd.read_csv(args.output)
        previous_lookup = {
            str(row.review_id): row._asdict()
            for row in previous.itertuples(index=False)
        }

    metadata = pd.read_csv(DATA / "brand_metadata.csv")
    reviews = json.loads(REVIEW_MANIFEST.read_text(encoding="utf-8"))["records"]
    review_lookup = {row["review_id"]: row for row in reviews}
    v2_entries = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))["entries"]

    metadata_lookup = {
        (str(row.category), normalised(row.brand_name)): row
        for row in metadata.itertuples(index=False)
    }
    rows = []
    for entry in v2_entries:
        key = (entry["category"], normalised(entry["brand_name"]))
        metadata_row = metadata_lookup.get(key)
        review_rows = [review_lookup[review_id] for review_id in entry.get("review_ids", [entry["review_id"]])]
        feedback = " ".join(
            str(value).strip()
            for review in review_rows
            for value in (review.get("reviewer_note", ""), review.get("resolved_action", ""))
            if str(value).strip()
        )
        current = text_card(pd.Series(metadata_row._asdict())) if metadata_row is not None else "No production description."
        image_path = APP_ASSETS / Path(entry["ios_v2_path"]).name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        rows.append({
            "review_id": entry["review_id"],
            "brand_name": entry["brand_name"],
            "category": entry["category"],
            "image_path": str(image_path.relative_to(ROOT)),
            "metadata_missing": metadata_row is None,
            "current_description": current,
            "review_feedback": feedback or current,
            "aesthetic_keywords": "" if metadata_row is None else metadata_row.aesthetic_keywords,
            "silhouettes": "" if metadata_row is None else metadata_row.silhouettes,
            "materials": "" if metadata_row is None else metadata_row.materials,
            "palette": "" if metadata_row is None else metadata_row.palette,
        })

    frame = pd.DataFrame(rows)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
    model = CLIPModel.from_pretrained(MODEL_NAME, local_files_only=True)
    model.eval()
    current_vectors = encode_texts(model, processor, frame["current_description"].tolist())
    feedback_vectors = encode_texts(model, processor, frame["review_feedback"].tolist())
    selected_images = encode_images(model, processor, [ROOT / path for path in frame["image_path"]])
    frame["current_image_similarity"] = np.sum(selected_images * current_vectors, axis=1)
    frame["feedback_image_similarity"] = np.sum(selected_images * feedback_vectors, axis=1)
    frame["feedback_advantage"] = frame["feedback_image_similarity"] - frame["current_image_similarity"]
    frame["current_similarity_percentile"] = frame["current_image_similarity"].rank(pct=True)
    frame["audit_priority"] = (
        (1.0 - frame["current_similarity_percentile"]) * 0.65
        + frame["feedback_advantage"].clip(lower=0) * 3.5
    )
    baseline_descriptions = []
    baseline_similarities = []
    for row in frame.itertuples(index=False):
        previous = previous_lookup.get(str(row.review_id), {})
        baseline_descriptions.append(
            previous.get("baseline_description", previous.get("current_description", row.current_description))
        )
        baseline_similarities.append(
            previous.get(
                "baseline_current_image_similarity",
                previous.get("current_image_similarity", row.current_image_similarity),
            )
        )
    frame["baseline_description"] = baseline_descriptions
    frame["baseline_current_image_similarity"] = baseline_similarities
    frame["description_changed"] = frame["current_description"].ne(frame["baseline_description"])
    frame["similarity_change"] = (
        frame["current_image_similarity"] - frame["baseline_current_image_similarity"]
    )
    frame = frame.sort_values(
        ["audit_priority", "feedback_advantage", "current_image_similarity"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, float_format="%.6f")
    print(f"Wrote {len(frame)} V2 alignment rows to {args.output.relative_to(ROOT)}")
    print(frame[["brand_name", "category", "current_image_similarity", "feedback_advantage", "audit_priority"]].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
