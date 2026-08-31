from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


STOPWORDS = {
    "and", "with", "the", "for", "from", "into", "that", "this", "brand", "brands",
    "modern", "contemporary", "classic", "premium", "easy", "high", "low", "medium",
    "mix", "led", "ready", "wear", "detail", "oriented", "pieces", "tones", "fabric",
    "cotton", "wool", "silk", "linen", "leather", "jersey", "viscose", "polyester",
    "black", "white", "ecru", "navy", "grey", "gray", "cream", "ivory", "blue", "brown",
    "gold", "silver", "bags", "bag", "shoes", "shoe", "clothes", "jewellery", "jewelry",
}


CLUSTER_TITLES = {
    ("clothes", 0): "Relaxed Basics",
    ("clothes", 1): "Printed Romantic Daywear",
    ("clothes", 2): "Polished Leather Tailoring",
    ("clothes", 3): "Western Casual Denim",
    ("clothes", 4): "Resort Swim",
    ("clothes", 5): "Glam Occasion Dresses",
    ("clothes", 6): "Soft Knitwear",
    ("clothes", 7): "Everyday Soft Tailoring",
    ("clothes", 8): "Soft Contemporary Dressing",
    ("clothes", 9): "Technical Utility",
    ("clothes", 10): "Bohemian Craft",
    ("clothes", 11): "Denim Jeans",
    ("clothes", 12): "Avant-Garde Layered Tailoring",
    ("clothes", 13): "Skate Streetwear",
    ("clothes", 14): "Graphic Tee Streetwear",
    ("clothes", 15): "Technical Utility Outerwear",
    ("clothes", 16): "Quiet Minimalism",
    ("clothes", 17): "Romantic Lingerie",
    ("shoes", 0): "Embellished Occasion Heels",
    ("shoes", 1): "Everyday Leather Staples",
    ("shoes", 2): "Formal Leather Loafers",
    ("shoes", 3): "Graphic Chunky Footwear",
    ("shoes", 4): "Minimal Ballet Flats",
    ("shoes", 5): "Artisanal Woven Sandals",
    ("shoes", 6): "Trend Statement Shoes",
    ("shoes", 7): "Outdoor Sport Footwear",
    ("shoes", 8): "Refined Sculptural Shoes",
    ("shoes", 9): "Resort Beach Sandals",
    ("shoes", 10): "Artisanal Sculpted Leather",
    ("shoes", 11): "Playful Ballet Flats",
    ("shoes", 12): "Artful Sculptural Footwear",
    ("shoes", 13): "Skate Shoes",
    ("shoes", 14): "Utility Comfort Clogs",
    ("shoes", 15): "Refined Sculptural Heels",
    ("shoes", 16): "Nautical Leather Loafers",
    ("shoes", 17): "Retro Clean Trainers",
    ("shoes", 18): "Heritage Clogs",
    ("shoes", 19): "Preppy Loafers and Pumps",
    ("shoes", 20): "Heritage Rugged Leather Shoes",
    ("shoes", 21): "Minimal Comfort Footwear",
    ("shoes", 22): "Relaxed Resort Sandals",
    ("shoes", 23): "Technical Running Shoes",
    ("bags", 0): "Heritage Leather Carryalls",
    ("bags", 1): "Graphic Urban Crossbodies",
    ("bags", 2): "Minimal Structured Crossbodies",
    ("bags", 3): "Functional Technical Totes",
    ("bags", 4): "Experimental Sculptural Bags",
    ("bags", 5): "Refined Sculptural Handbags",
    ("bags", 6): "Sport Utility Backpacks",
    ("bags", 7): "Ethical Craft Carryalls",
    ("bags", 8): "Woven Natural Bags",
    ("bags", 9): "Occasion Crystal Pouches",
    ("bags", 10): "Resort Beach Totes",
    ("bags", 11): "Playful Soft Shoulder Bags",
    ("bags", 12): "Polished Structured Totes",
    ("bags", 13): "Minimal Vegan Bags",
    ("bags", 14): "Feminine Structured Shoulder Bags",
    ("bags", 15): "Artisanal Resort Carryalls",
    ("bags", 16): "Everyday Pouch Bags",
    ("jewellery", 0): "Pearl Fine Jewellery",
    ("jewellery", 1): "Delicate Fine Chains",
    ("jewellery", 2): "Bold Sculptural Hoops",
    ("jewellery", 3): "Polished Pearl Jewellery",
    ("jewellery", 4): "Bohemian Beaded Jewellery",
    ("jewellery", 5): "Talismanic Enamel Jewellery",
    ("jewellery", 6): "Minimal Sculptural Jewellery",
    ("jewellery", 7): "Preppy Pearl Accessories",
    ("jewellery", 8): "Utility Metal Charms",
    ("jewellery", 9): "Talismanic Gemstone Jewellery",
    ("jewellery", 10): "Organic Sculptural Rings",
    ("jewellery", 11): "Sculptural Chain Jewellery",
    ("jewellery", 12): "Opulent Gemstone Fine Jewellery",
    ("jewellery", 13): "Resort Body Jewellery",
    ("jewellery", 14): "Crystal Occasion Jewellery",
    ("jewellery", 15): "Playful Charm Jewellery",
}


def build_category_dataset(source_rows: pd.DataFrame, embeddings: np.ndarray, category: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Group one category so each brand has one row and one vector."""
    cat_rows = source_rows[source_rows["category"].eq(category)].copy().reset_index(drop=True)
    grouped_rows = []
    grouped_vectors = []
    for brand_name, group in cat_rows.groupby("brand_name", sort=True):
        idx = group["row_id"].to_numpy()
        grouped_vectors.append(normalize(embeddings[idx].mean(axis=0, keepdims=True))[0])
        grouped_rows.append({
            "brand_name": brand_name,
            "category": category,
            "official_website": group["official_website"].dropna().astype(str).iloc[0] if "official_website" in group and group["official_website"].notna().any() else "",
            "aesthetic_keywords": ", ".join(group["aesthetic_keywords"].dropna().astype(str).unique()),
            "silhouettes": ", ".join(group["silhouettes"].dropna().astype(str).unique()),
            "materials": ", ".join(group["materials"].dropna().astype(str).unique()),
            "palette": ", ".join(group["palette"].dropna().astype(str).unique()),
            "moodboard": ", ".join(sorted(group.get("moodboard", pd.Series(dtype=str)).dropna().astype(str).unique())),
            "n_rows": len(group),
        })
    return pd.DataFrame(grouped_rows).reset_index(drop=True), np.vstack(grouped_vectors).astype("float32")


def minmax(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Scale a score column from zero to one."""
    values = series.astype(float)
    span = values.max() - values.min()
    if span == 0:
        return pd.Series(0.5, index=series.index)
    scaled = (values - values.min()) / span
    return scaled if higher_is_better else 1 - scaled


def tokenize_style_text(text: str) -> list[str]:
    """Split text into useful style words."""
    words = re.findall(r"[a-z][a-z]+", str(text).lower().replace("-", " "))
    return [word for word in words if word not in STOPWORDS and len(word) > 2]


def split_terms(text: str) -> list[str]:
    """Split comma-separated style fields into clean terms."""
    terms = []
    for chunk in str(text).split(","):
        term = chunk.strip().lower().replace("-", " ")
        if term and term not in {"nan", "none"}:
            terms.append(term)
    return terms


def representative_brands(brands: pd.DataFrame, X: np.ndarray, labels: np.ndarray, top_n: int = 6) -> pd.DataFrame:
    """Find brands closest to each cluster centre."""
    labels = np.asarray(labels)
    rows = []
    for label in sorted(set(labels)):
        if label == -1:
            continue
        idx = np.where(labels == label)[0]
        centroid = normalize(X[idx].mean(axis=0, keepdims=True))[0]
        sims = cosine_similarity(X[idx], centroid.reshape(1, -1)).ravel()
        for local_i in np.argsort(-sims)[:top_n]:
            brand_i = idx[local_i]
            rows.append({
                "cluster": int(label),
                "brand_name": brands.loc[brand_i, "brand_name"],
                "centroid_similarity": float(sims[local_i]),
            })
    return pd.DataFrame(rows)


def separate_text_labels(ax, texts: list[object], pad: float = 3.0, max_iter: int = 160) -> None:
    """Nudge plot labels apart so they do not sit on top of each other."""
    if len(texts) < 2:
        return
    fig = ax.figure
    fig.canvas.draw()
    for _ in range(max_iter):
        renderer = fig.canvas.get_renderer()
        bboxes = [text.get_window_extent(renderer).expanded(1.04, 1.08) for text in texts]
        shifts = np.zeros((len(texts), 2), dtype=float)
        moved = False
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if not bboxes[i].overlaps(bboxes[j]):
                    continue
                x_overlap = min(bboxes[i].x1, bboxes[j].x1) - max(bboxes[i].x0, bboxes[j].x0)
                y_overlap = min(bboxes[i].y1, bboxes[j].y1) - max(bboxes[i].y0, bboxes[j].y0)
                if x_overlap <= 0 or y_overlap <= 0:
                    continue
                ci = np.array([(bboxes[i].x0 + bboxes[i].x1) / 2, (bboxes[i].y0 + bboxes[i].y1) / 2])
                cj = np.array([(bboxes[j].x0 + bboxes[j].x1) / 2, (bboxes[j].y0 + bboxes[j].y1) / 2])
                delta = ci - cj
                if np.linalg.norm(delta) < 1e-6:
                    angle = (i + 1) * 2.399963229728653
                    delta = np.array([np.cos(angle), np.sin(angle)])
                if x_overlap < y_overlap:
                    direction = 1 if delta[0] >= 0 else -1
                    shift = np.array([direction * (x_overlap / 2 + pad), 0.0])
                else:
                    direction = 1 if delta[1] >= 0 else -1
                    shift = np.array([0.0, direction * (y_overlap / 2 + pad)])
                shifts[i] += shift
                shifts[j] -= shift
                moved = True
        if not moved:
            break
        ax_bbox = ax.get_window_extent(renderer)
        for idx, text in enumerate(texts):
            if not np.any(shifts[idx]):
                continue
            current = ax.transData.transform(text.get_position())
            current += np.clip(shifts[idx], -28, 28)
            half_w = bboxes[idx].width / 2
            half_h = bboxes[idx].height / 2
            current[0] = np.clip(current[0], ax_bbox.x0 + half_w + 2, ax_bbox.x1 - half_w - 2)
            current[1] = np.clip(current[1], ax_bbox.y0 + half_h + 2, ax_bbox.y1 - half_h - 2)
            text.set_position(ax.transData.inverted().transform(current))
        fig.canvas.draw()
