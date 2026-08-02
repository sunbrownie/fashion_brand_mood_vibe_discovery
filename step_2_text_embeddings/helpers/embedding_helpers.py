# Helper code for the embeddings notebook.

from pathlib import Path

import numpy as np
import pandas as pd

CWD = Path.cwd().resolve()
ROOT = CWD if (CWD / "final_dataset").exists() else CWD.parent

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
if not OUT_DIR.exists():
    OUT_DIR = CWD
EMBEDDINGS_PATH = OUT_DIR / "brand_embeddings.npz"
METADATA_PATH = OUT_DIR / "brand_metadata.csv"

RNG_SEED = 7


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
