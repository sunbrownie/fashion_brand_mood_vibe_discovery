#!/usr/bin/env python3
"""Build Ellina swimwear/lingerie correction review PDF with old/new/Google evidence."""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "tmp" / "swimwear_lingerie_old"
NEW = ROOT / "moodboards_creation" / "moodboards" / "swimwear_lingerie"
REFERENCES = ROOT / "step_3_moodboards" / "review" / "reference_screenshots" / "swimwear_lingerie"
DATA = ROOT / "recommender" / "hosting_bundle" / "data"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "swimwear_lingerie_review_round.pdf"

PAGE_W, PAGE_H = landscape(A4)
INK = HexColor("#17243A")
MUTED = HexColor("#657083")
PAPER = HexColor("#F7F1E7")
WHITE = HexColor("#FFFEFB")
PINK = HexColor("#E66F89")
GREEN = HexColor("#6C997B")
PALE_PINK = HexColor("#FCE8ED")
PALE_GREEN = HexColor("#E7F0E9")


@dataclass(frozen=True)
class ReviewItem:
    brand: str
    slug: str
    comment: str
    action: str
    summary: str


ITEMS = (
    ReviewItem("Acacia", "acacia", "remove the cardigan", "UPDATED", "Removed generic cardigan/knitwear; retained the sun-washed bohemian swim identity."),
    ReviewItem("Araks", "araks", "3 duplicate images.", "UPDATED", "Replaced repeated swimsuit, satin and brief images with distinct minimal pieces."),
    ReviewItem("Baserange", "baserange", "change the top middle image - weird for udnerpants", "UPDATED", "Replaced the awkward top-centre crop with a clear flat-lay brief. Production wording uses “underwear”."),
    ReviewItem("Fruity Booty", "fruity_booty", "2 duplicate images", "UPDATED", "Removed repeated red lingerie and striped shorts; added distinct playful pieces."),
    ReviewItem("Intimissimi", "intimissimi", "remove words", "UPDATED", "Removed every visible word and label while preserving silk, lace and Italian refinement."),
    ReviewItem("Left On Friday", "left_on_friday", "same image twice", "UPDATED", "Replaced the repeated red-suit image with a separate performance action look."),
    ReviewItem("Natori", "natori", "same image twice", "UPDATED", "Removed the repeated blush-slip model and added distinct embroidered nightwear."),
    ReviewItem("Only Hearts", "only_hearts", "same image twice", "UPDATED", "Removed the repeated lounge model and kept a soft product-led romantic identity."),
    ReviewItem("Perle by Lola", "perle_by_lola", "remove, very limited swimwear, no lingerie", "REMOVED", "Removed from the active swimwear/lingerie catalog, embeddings and cluster assignments."),
    ReviewItem("Von Dutch", "von_dutch", "remove - off brand right now, and they do limited swimwear", "REMOVED", "Removed from the active swimwear/lingerie catalog, embeddings and cluster assignments."),
    ReviewItem("Calvin Klein", "calvin_klein", "To add: Calvin Klein", "ADDED", "Added minimalist underwear and swim moodboard, text embedding and cluster assignment."),
    ReviewItem("La Perla", "la_perla", "To add: La Perla", "ADDED", "Added couture lace/silk moodboard, text embedding and cluster assignment."),
    ReviewItem("Calzedonia", "calzedonia", "To add: Calzedonia", "ADDED", "Added trend-led Italian swim moodboard, text embedding and cluster assignment."),
    ReviewItem("Billabong", "billabong", "To add: Billabong", "ADDED", "Added surf-performance swim moodboard, text embedding and cluster assignment."),
    ReviewItem("Roxy", "roxy", "To add: Roxy", "ADDED", "Added women’s surf swim moodboard, text embedding and cluster assignment."),
    ReviewItem("Wolford", "wolford", "To add: Wolford", "ADDED", "Added technical bodywear/hosiery moodboard, text embedding and cluster assignment."),
)


def draw_wrapped(c: canvas.Canvas, text: str, x: float, y: float, width_chars: int, leading: float, font: str = "Helvetica", size: float = 10, color=INK) -> float:
    c.setFont(font, size)
    c.setFillColor(color)
    for line in textwrap.wrap(text, width=width_chars) or [""]:
        c.drawString(x, y, line)
        y -= leading
    return y


def draw_fitted_image(c: canvas.Canvas, path: Path, x: float, y: float, w: float, h: float) -> None:
    c.setFillColor(WHITE)
    c.roundRect(x, y, w, h, 8, fill=1, stroke=0)
    if not path.is_file():
        c.setFillColor(MUTED)
        c.setFont("Helvetica-Bold", 15)
        c.drawCentredString(x + w / 2, y + h / 2 + 6, "Not present")
        c.setFont("Helvetica", 10)
        c.drawCentredString(x + w / 2, y + h / 2 - 12, "No moodboard in this version")
        return
    with Image.open(path) as image:
        iw, ih = image.size
    scale = min((w - 12) / iw, (h - 12) / ih)
    dw, dh = iw * scale, ih * scale
    c.drawImage(ImageReader(str(path)), x + (w - dw) / 2, y + (h - dh) / 2, width=dw, height=dh, preserveAspectRatio=True, mask="auto")


def status_style(action: str):
    if action == "REMOVED":
        return PALE_PINK, PINK
    return PALE_GREEN, GREEN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(DATA / "brand_metadata.csv")
    assignments = pd.read_csv(DATA / "ios_brand_cluster_assignments.csv")
    meta = {
        str(row.brand_name).casefold(): row
        for row in metadata[metadata["category"].eq("swimwear_lingerie")].itertuples()
    }
    assigned = {
        str(row.brand_name).casefold(): row
        for row in assignments[assignments["category"].eq("swimwear_lingerie")].itertuples()
    }

    c = canvas.Canvas(str(args.output), pagesize=landscape(A4), pageCompression=1)
    c.setTitle("Swimwear & Lingerie — Ellina Review Corrections")
    c.setAuthor("ModeNova")

    c.setFillColor(PAPER)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 31)
    c.drawString(46, PAGE_H - 90, "Swimwear & Lingerie")
    c.setFillColor(PINK)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(46, PAGE_H - 126, "Ellina review — old, final and Google reference")
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 12)
    c.drawString(46, PAGE_H - 164, f"Generated {date.today().isoformat()} · 8 updated · 2 removed · 6 added")
    draw_wrapped(c, "Each page preserves Ellina’s latest comment, shows the previous and current state beside a Google Images reference, and records the text/cluster result used by the iOS and Hugging Face builds.", 46, PAGE_H - 204, 92, 17, size=12, color=INK)
    c.setFillColor(PALE_GREEN)
    c.roundRect(46, 66, PAGE_W - 92, 110, 14, fill=1, stroke=0)
    c.setFillColor(GREEN)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(66, 145, "Published state")
    c.setFillColor(INK)
    c.setFont("Helvetica", 11)
    c.drawString(66, 119, f"Catalog rows: {len(metadata):,}  ·  Active taste assignments: {len(assignments):,}  ·  Text embedding dimensions: 768")
    c.drawString(66, 96, "Only the final active swimwear/lingerie versions are included in the application bundles.")
    c.showPage()

    for page_number, item in enumerate(ITEMS, start=2):
        c.setFillColor(PAPER)
        c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 25)
        c.drawString(34, PAGE_H - 42, item.brand)
        bg, fg = status_style(item.action)
        c.setFillColor(bg)
        c.roundRect(PAGE_W - 132, PAGE_H - 58, 96, 26, 13, fill=1, stroke=0)
        c.setFillColor(fg)
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(PAGE_W - 84, PAGE_H - 49, item.action)

        c.setFillColor(PALE_PINK)
        c.roundRect(34, PAGE_H - 112, PAGE_W - 68, 48, 9, fill=1, stroke=0)
        c.setFillColor(PINK)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(48, PAGE_H - 82, "ELLINA’S LATEST COMMENT")
        c.setFillColor(INK)
        c.setFont("Helvetica-Oblique", 11)
        c.drawString(48, PAGE_H - 101, f'“{item.comment}”')

        image_y, image_h = 126, 322
        image_w = (PAGE_W - 104) / 3
        old_path = OLD / f"{item.slug}.jpg"
        new_path = NEW / f"{item.slug}.jpg"
        reference_path = REFERENCES / f"{item.slug}.png"
        if item.action == "ADDED":
            old_path = ROOT / "tmp" / "pdfs" / "not_present.jpg"
        if item.action == "REMOVED":
            new_path = ROOT / "tmp" / "pdfs" / "removed.jpg"
        c.setFillColor(MUTED)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(34, image_y + image_h + 10, "PREVIOUS")
        c.drawString(52 + image_w, image_y + image_h + 10, "CURRENT / FINAL")
        c.drawString(70 + 2 * image_w, image_y + image_h + 10, "GOOGLE IMAGES REFERENCE")
        draw_fitted_image(c, old_path, 34, image_y, image_w, image_h)
        draw_fitted_image(c, new_path, 52 + image_w, image_y, image_w, image_h)
        draw_fitted_image(c, reference_path, 70 + 2 * image_w, image_y, image_w, image_h)

        c.setFillColor(WHITE)
        c.roundRect(34, 36, PAGE_W - 68, 76, 9, fill=1, stroke=0)
        y = draw_wrapped(c, item.summary, 48, 92, 112, 13, font="Helvetica-Bold", size=9.5)
        key = item.brand.casefold()
        if key in assigned:
            row = assigned[key]
            cluster_line = f"Archetype: {row.public_display_name}"
            draw_wrapped(c, cluster_line, 48, y - 2, 118, 12, size=8.5, color=MUTED)
        elif item.action == "REMOVED":
            draw_wrapped(c, "No active embedding or cluster assignment remains for this category.", 48, y - 2, 118, 12, size=8.5, color=MUTED)
        if key in meta:
            row = meta[key]
            draw_wrapped(c, f"Text: {row.aesthetic_keywords} · {row.silhouettes}", 48, 48, 130, 10, size=7.4, color=MUTED)

        c.setFillColor(MUTED)
        c.setFont("Helvetica", 7.5)
        c.drawRightString(PAGE_W - 34, 18, f"ModeNova · Swimwear/Lingerie Review · {page_number}/{len(ITEMS) + 1}")
        c.showPage()

    c.save()
    print(args.output)


if __name__ == "__main__":
    main()
