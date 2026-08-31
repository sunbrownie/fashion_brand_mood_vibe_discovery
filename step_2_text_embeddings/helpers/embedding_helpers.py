# Helper code for the embeddings notebook.

from pathlib import Path

import numpy as np
import pandas as pd

CWD = Path.cwd().resolve()
ROOT = CWD.parent if CWD.name == "step_2_text_embeddings" else CWD

# model
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# data
DATA_DIR = ROOT / "final_dataset"
CSV_FILES = {
    "clothes": DATA_DIR / "brands_clothes.csv",
    "shoes": DATA_DIR / "brands_shoes.csv",
    "bags": DATA_DIR / "brands_bags.csv",
    "jewellery": DATA_DIR / "brands_jewellery.csv",
}

# output
OUT_DIR = ROOT / "step_2_text_embeddings"
EMBEDDINGS_PATH = OUT_DIR / "brand_embeddings.npz"
METADATA_PATH = OUT_DIR / "brand_metadata.csv"

RNG_SEED = 7

THESIS_DATASET_ROWS = [
    {
        "brand_name": "ANNA + NINA",
        "category": "clothes",
        "official_website": "https://anna-nina.nl",
        "aesthetic_keywords": "vintage-inspired, delicate, gemstone-focused, eclectic, artisanal",
        "silhouettes": "rings, necklaces, earrings, charms",
        "materials": "14k and 9k gold, gemstones, diamonds",
        "palette": "jewel tones, gold, ruby, neutral",
        "male": 0.0,
        "female": 1.0,
        "children": 0.0,
        "price": "upper price point",
    },
    {
        "brand_name": "Paris Texas",
        "category": "clothes",
        "official_website": "https://paristexasco.com",
        "aesthetic_keywords": "sleek, sensual, bold, sculpted, statement",
        "silhouettes": "sculpted boots, heeled silhouettes, sleek footwear",
        "materials": "polished leather, suede",
        "palette": "black, red, tan, brights",
        "male": 1.0,
        "female": 1.0,
        "children": 0.0,
        "price": "upper price point",
    },
    {
        "brand_name": "ANNA + NINA",
        "category": "shoes",
        "official_website": "https://anna-nina.nl",
        "aesthetic_keywords": "vintage-inspired, delicate, gemstone-focused, eclectic, artisanal",
        "silhouettes": "loafers, ballet flats, sandals, ankle boots, clean sneakers",
        "materials": "14k and 9k gold, gemstones, diamonds",
        "palette": "jewel tones, gold, ruby, neutral",
    },
    {
        "brand_name": "ANNA + NINA",
        "category": "bags",
        "official_website": "https://anna-nina.nl",
        "aesthetic_keywords": "vintage-inspired, delicate, gemstone-focused, eclectic, artisanal",
        "silhouettes": "shoulder bags, compact crossbodies, structured totes, soft pouches",
        "materials": "14k and 9k gold, gemstones, diamonds",
        "palette": "jewel tones, gold, ruby, neutral",
    },
]


def align_to_thesis_dataset(rows: pd.DataFrame) -> pd.DataFrame:
    """Applies the small row fix needed for the thesis embedding snapshot."""
    rows = rows.copy()
    rows = rows[~(rows["category"].eq("shoes") & rows["brand_name"].eq("Paris Texas"))]

    for patch_row in THESIS_DATASET_ROWS:
        has_row = rows["category"].eq(patch_row["category"]) & rows["brand_name"].eq(patch_row["brand_name"])
        if not has_row.any():
            rows = pd.concat([rows, pd.DataFrame([patch_row])], ignore_index=True, sort=False)

    category_order = {category: index for index, category in enumerate(CSV_FILES)}
    rows["_category_order"] = rows["category"].map(category_order)
    rows["_brand_order"] = rows["brand_name"].astype(str).str.casefold()
    rows = rows.sort_values(["_category_order", "_brand_order"])
    rows = rows.drop(columns=["_category_order", "_brand_order"])
    return rows.reset_index(drop=True)


def to_prompt(row: pd.Series) -> str:
    """Takes a brand row and returns one text prompt by joining its vibe fields."""
    def clean(val):
        """Takes a value and returns clean text by blanking missing values."""
        return str(val).strip() if pd.notna(val) and str(val).strip() else ""

    parts = []
    if kw := clean(row.get("aesthetic_keywords")):
        parts.append(f"Aesthetic: {kw}.")
    if sil := clean(row.get("silhouettes")):
        parts.append(f"Silhouettes: {sil}.")
    if mat := clean(row.get("materials")):
        parts.append(f"Materials: {mat}.")
    if pal := clean(row.get("palette")):
        parts.append(f"Palette: {pal}.")
    return " ".join(parts)




def nearest_neighbours(brand_name: str, category: str | None = None, k: int = 5) -> pd.DataFrame:
    """Takes a brand name and returns nearest brands by comparing embeddings."""
    mask = meta_loaded["brand_name"] == brand_name
    if category is not None:
        mask &= meta_loaded["category"] == category

    idx = meta_loaded.index[mask]
    if len(idx) == 0:
        label = f"{brand_name!r}" if category is None else f"{brand_name!r} in category {category!r}"
        raise ValueError(f"{label} not found in metadata")
    if len(idx) > 1:
        categories = sorted(meta_loaded.loc[idx, "category"].unique())
        raise ValueError(
            f"{brand_name!r} appears in multiple categories: {categories}. "
            "Pass category=... to choose one."
        )

    query_idx = idx[0]
    query_vec = emb_loaded[query_idx]
    scores = emb_loaded @ query_vec
    # Exclude only the exact query row; keep same-brand rows from other categories visible.
    scores[query_idx] = -1.0
    top_idx = np.argsort(scores)[::-1][:k]
    result = meta_loaded.iloc[top_idx][["brand_name", "category", "aesthetic_keywords"]].copy()
    result["cosine_sim"] = scores[top_idx].round(4)
    return result.reset_index(drop=True)
