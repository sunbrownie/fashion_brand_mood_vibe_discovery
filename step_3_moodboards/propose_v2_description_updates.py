#!/usr/bin/env python3
"""Propose structured text cards from the accepted V2 moodboard visuals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from audit_v2_descriptions import APP_ASSETS, MODEL_NAME


ROOT = Path(__file__).resolve().parents[1]
V2_MANIFEST = APP_ASSETS / "v2_manifest.json"
AUDIT = ROOT / "step_3_moodboards" / "review" / "v2_description_alignment_audit.csv"
OUTPUT = ROOT / "step_3_moodboards" / "review" / "v2_description_update_proposals.csv"

AESTHETICS = (
    "minimal", "maximalist", "romantic", "bohemian", "streetwear", "sporty", "technical", "preppy",
    "heritage", "avant-garde", "sculptural", "playful", "whimsical", "surreal", "gothic", "punk",
    "grunge", "retro", "Y2K", "seventies", "eighties", "nineties", "resort", "utilitarian", "workwear",
    "outdoorsy", "motorcycle", "western", "collegiate", "tailored", "sensual", "glamorous", "party",
    "artisanal", "handcrafted", "folkloric", "eclectic", "colourful", "monochrome", "quiet luxury",
    "modernist", "futuristic", "architectural", "casual", "polished", "classic", "rebellious", "youthful",
    "elegant", "quirky", "graphic", "logo-led", "print-led", "denim-led", "knitwear-led", "cottagecore",
    "soft feminine", "androgynous", "clean-lined", "deconstructed", "upcycled", "sport-luxe", "downtown",
)

SILHOUETTES = {
    "clothes": (
        "tailored suits", "oversized blazers", "wide-leg trousers", "cargo trousers", "midi dresses", "maxi dresses",
        "mini dresses", "slip dresses", "sheer dresses", "evening gowns", "corset tops", "graphic T-shirts",
        "hoodies and sweatshirts", "cardigans", "chunky sweaters", "playful knitwear", "leather jackets",
        "motorcycle jackets", "technical shell jackets", "puffer jackets", "trench coats", "denim jackets",
        "patchwork denim", "pleated skirts", "mini skirts", "flowing skirts", "co-ordinated sets", "tracksuits",
        "activewear sets", "workwear separates", "romantic blouses", "asymmetric tops", "sculptural tailoring",
    ),
    "shoes": (
        "low-top sneakers", "high-top sneakers", "retro court sneakers", "technical running shoes", "trail shoes",
        "football boots", "platform shoes", "chunky boots", "motorcycle boots", "ankle boots", "knee-high boots",
        "snowboard boots", "loafers", "ballet flats", "pumps", "slingback heels", "sculptural heels",
        "strappy sandals", "sport sandals", "mules", "clogs", "espadrilles", "driving shoes", "western boots",
    ),
    "bags": (
        "structured tote bags", "soft tote bags", "shoulder bags", "crescent shoulder bags", "hobo bags",
        "baguette bags", "top-handle bags", "bucket bags", "crossbody bags", "chain bags", "mini bags", "clutches",
        "backpacks", "belt bags", "woven bags", "novelty bags", "sculptural bags", "bowling bags", "fringed bags",
    ),
    "jewellery": (
        "hoop earrings", "drop earrings", "statement earrings", "stud earrings", "pendant necklaces", "chain necklaces",
        "beaded necklaces", "charm necklaces", "stacking rings", "statement rings", "cuff bracelets", "charm bracelets",
        "chain bracelets", "brooches", "body jewellery", "pearls", "sculptural metal jewellery",
    ),
}

MATERIALS = (
    "smooth leather", "grained leather", "patent leather", "perforated leather", "suede", "canvas", "cotton jersey",
    "technical nylon", "mesh", "rubber", "denim", "wool", "chunky knit", "crochet", "silk", "satin", "velvet",
    "lace", "tulle", "sequins", "metallic fabric", "silver-tone metal", "gold-tone metal", "crystals", "pearls",
    "resin", "enamel", "raffia", "straw", "beads", "embroidery", "tartan fabric", "recycled textiles",
)

PALETTE = (
    "black", "white", "cream", "beige", "tan", "chocolate brown", "grey", "navy", "sky blue", "cobalt blue",
    "red", "burgundy", "pink", "orange", "yellow", "lime green", "forest green", "teal", "purple", "silver",
    "gold", "multicolour", "earthy neutrals", "soft pastels", "jewel tones", "neon accents",
)

MOTIFS = (
    "plaid and tartan", "floral prints", "animal prints", "checkerboard", "stripes", "visible logos", "heart motifs",
    "crescent moon motifs", "star motifs", "fruit motifs", "abstract prints", "patchwork", "studs and grommets",
    "chains and hardware", "fringe", "embroidery", "bows", "ruffles", "quilting", "perforated patterns",
    "graphic slogans", "surreal motifs", "colour blocking", "distressed finishes", "charms", "beading",
)


def encode_texts(model: CLIPModel, processor: CLIPProcessor, labels: tuple[str, ...]) -> np.ndarray:
    prompts = [f"a fashion moodboard featuring {label}" for label in labels]
    batch = processor(text=prompts, padding=True, truncation=True, return_tensors="pt")
    with torch.inference_mode():
        values = model.get_text_features(**batch)
        if not isinstance(values, torch.Tensor):
            values = values.pooler_output
        values = torch.nn.functional.normalize(values, dim=-1)
    return values.cpu().numpy().astype("float32")


def encode_images(model: CLIPModel, processor: CLIPProcessor, paths: list[Path]) -> np.ndarray:
    output = []
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


def top_labels(image: np.ndarray, labels: tuple[str, ...], vectors: np.ndarray, count: int) -> list[str]:
    scores = vectors @ image
    return [labels[index] for index in np.argsort(-scores)[:count]]


def card(aesthetic_keywords: str, silhouettes: str, materials: str, palette: str) -> str:
    return f"Aesthetic: {aesthetic_keywords}. Silhouettes: {silhouettes}. Materials: {materials}. Palette: {palette}."


def main() -> None:
    audit = pd.read_csv(AUDIT)
    manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))["entries"]
    path_lookup = {
        (entry["brand_name"], entry["category"]): APP_ASSETS / Path(entry["ios_v2_path"]).name
        for entry in manifest
    }
    paths = [path_lookup[(row.brand_name, row.category)] for row in audit.itertuples()]

    processor = CLIPProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
    model = CLIPModel.from_pretrained(MODEL_NAME, local_files_only=True)
    model.eval()
    images = encode_images(model, processor, paths)
    vocab_vectors = {
        "aesthetic": encode_texts(model, processor, AESTHETICS),
        "materials": encode_texts(model, processor, MATERIALS),
        "palette": encode_texts(model, processor, PALETTE),
        "motifs": encode_texts(model, processor, MOTIFS),
    }
    silhouette_vectors = {
        category: encode_texts(model, processor, labels)
        for category, labels in SILHOUETTES.items()
    }

    proposed_rows = []
    for index, row in enumerate(audit.itertuples()):
        image = images[index]
        aesthetic_terms = top_labels(image, AESTHETICS, vocab_vectors["aesthetic"], 4)
        motif_terms = top_labels(image, MOTIFS, vocab_vectors["motifs"], 2)
        silhouette_terms = top_labels(image, SILHOUETTES[row.category], silhouette_vectors[row.category], 5)
        material_terms = top_labels(image, MATERIALS, vocab_vectors["materials"], 4)
        palette_terms = top_labels(image, PALETTE, vocab_vectors["palette"], 5)
        proposed = {
            "aesthetic_keywords": ", ".join([*aesthetic_terms, *motif_terms]),
            "silhouettes": ", ".join(silhouette_terms),
            "materials": ", ".join(material_terms),
            "palette": ", ".join(palette_terms),
        }
        proposed_rows.append(proposed)

    proposed = pd.DataFrame(proposed_rows)
    proposed_cards = [card(**row) for row in proposed.to_dict("records")]
    proposed_vectors = encode_texts(model, processor, tuple(proposed_cards))
    proposed_scores = np.sum(images * proposed_vectors, axis=1)
    result = audit.copy()
    for column in proposed.columns:
        result[f"proposed_{column}"] = proposed[column]
    result["proposed_image_similarity"] = proposed_scores
    result["proposed_advantage"] = result["proposed_image_similarity"] - result["current_image_similarity"]
    result["recommended_update"] = (
        result["metadata_missing"].astype(bool)
        | ((result["current_image_similarity"] < 0.30) & (result["proposed_advantage"] >= 0.018))
    )
    result = result.sort_values(["recommended_update", "proposed_advantage"], ascending=[False, False])
    result.to_csv(OUTPUT, index=False, float_format="%.6f")
    selected = result[result["recommended_update"]]
    print(f"Proposed {len(selected)} metadata updates from {len(result)} accepted V2 moodboards.")
    print(selected[["brand_name", "category", "current_image_similarity", "proposed_image_similarity", "proposed_advantage"]].to_string(index=False))


if __name__ == "__main__":
    main()
