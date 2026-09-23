#!/usr/bin/env python3
"""Build the side-by-side review pack for Ellina's approved V2 pilots."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


PAGE_W, PAGE_H = landscape(A4)
INK = HexColor("#17243A")
MUTED = HexColor("#667085")
CREAM = HexColor("#F7F3EA")
PAPER = HexColor("#FFFEFB")
PINK = HexColor("#EAC1C8")
ACCENT = HexColor("#A23B55")
GREEN = HexColor("#3E7659")
LINE = HexColor("#D7D2C8")


PILOTS = [
    {
        "review_id": "ER-001",
        "brand": "Acne Studios",
        "category": "Clothes",
        "original": "moodboards_creation/moodboards/clothes/acne_studios.jpg",
        "updated": "ios/BrandMoodboardFinder/BrandMoodboardFinder/BrandMoodboards/v2__clothes__acne_studios.jpg",
        "feedback": (
            "Off vibe, other than the suit. The branded long sleeve is recognisable. "
            "Maybe a scarf. Skinny vs layered looks are a signature mix."
        ),
        "changes": [
            "Rebuilt around sharp tailoring and cool urban styling.",
            "Added a signature scarf and an explicit skinny-vs-layered contrast.",
            "Removed generic interiors, bean motifs and repeated outfit ideas.",
        ],
        "review": (
            "Approved for V2. The revised board has distinct looks, mixed casting and a tighter "
            "Acne-specific silhouette story."
        ),
    },
    {
        "review_id": "ER-236",
        "brand": "Arizona Love",
        "category": "Bags",
        "original": "moodboards_creation/moodboards/bags/arizona_love.jpg",
        "updated": "ios/BrandMoodboardFinder/BrandMoodboardFinder/BrandMoodboards/v2__bags__arizona_love.jpg",
        "feedback": (
            "Off vibe. Needs the signature bandana pattern. Overall vibe should be more "
            "cowboy-desert (Arizona) than beachy straw."
        ),
        "changes": [
            "Replaced the generic straw-and-beach direction with a desert setting.",
            "Made bandana textiles the main product language across distinct bag types.",
            "Added cowboy details while keeping the focus on bags.",
        ],
        "review": (
            "Approved for V2. The bandana construction and desert cues now answer the feedback "
            "directly, with no repeated product crop."
        ),
    },
    {
        "review_id": "ER-173",
        "brand": "Chanel",
        "category": "Shoes",
        "original": "moodboards_creation/moodboards/shoes/chanel.jpg",
        "updated": "ios/BrandMoodboardFinder/BrandMoodboardFinder/BrandMoodboards/v2__shoes__chanel.jpg",
        "feedback": (
            "Off vibe and bad collage - all a geometric grid with repeated images. Need signature "
            "ballet flats, tweed pumps and slingbacks."
        ),
        "changes": [
            "Replaced the repeated product grid with an editorial torn-paper composition.",
            "Added two-tone slingbacks, quilted ballet flats and tweed pumps.",
            "Kept all hero fragments footwear-specific and distinct.",
        ],
        "review": (
            "Approved for V2 after rejecting an earlier candidate. The final version contains seven "
            "distinct footwear fragments, no handbags and no repeated silhouette."
        ),
    },
    {
        "review_id": "ER-221",
        "brand": "Tory Burch",
        "category": "Jewellery",
        "original": "moodboards_creation/moodboards/jewellery/tory_burch.jpg",
        "updated": "ios/BrandMoodboardFinder/BrandMoodboardFinder/BrandMoodboards/v2__jewellery__tory_burch.jpg",
        "feedback": "Off vibe. The brand is less boho; include logo elements.",
        "changes": [
            "Shifted from Mediterranean-boho styling to polished, preppy navy and cream.",
            "Used cleaner gold forms and one controlled double-T pendant.",
            "Removed repeated logo motifs after an earlier candidate review.",
        ],
        "review": (
            "Approved for V2. The updated board is more composed and brand-specific while the "
            "single logo element avoids visual repetition."
        ),
    },
]


def wrap_lines(text: str, font: str, size: float, width: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        proposal = word if not current else f"{current} {word}"
        if stringWidth(proposal, font, size) <= width:
            current = proposal
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    font: str = "Helvetica",
    size: float = 9,
    leading: float = 12,
    color=INK,
    max_lines: int | None = None,
) -> float:
    lines = wrap_lines(text, font, size, width)
    if max_lines is not None:
        lines = lines[:max_lines]
    pdf.setFillColor(color)
    pdf.setFont(font, size)
    for line in lines:
        pdf.drawString(x, y, line)
        y -= leading
    return y


def draw_bullets(pdf: canvas.Canvas, items: list[str], x: float, y: float, width: float) -> float:
    for item in items:
        pdf.setFillColor(ACCENT)
        pdf.circle(x + 3, y + 3, 2, fill=1, stroke=0)
        y = draw_wrapped(pdf, item, x + 13, y, width - 13, size=8.4, leading=11) - 3
    return y


def draw_fitted_image(pdf: canvas.Canvas, path: Path, x: float, y: float, w: float, h: float) -> None:
    with PILImage.open(path) as image:
        iw, ih = image.size
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    ix, iy = x + (w - dw) / 2, y + (h - dh) / 2
    pdf.setFillColor(PAPER)
    pdf.roundRect(x, y, w, h, 8, fill=1, stroke=0)
    pdf.drawImage(str(path), ix, iy, width=dw, height=dh, preserveAspectRatio=True, mask="auto")
    pdf.setStrokeColor(LINE)
    pdf.roundRect(x, y, w, h, 8, fill=0, stroke=1)


def footer(pdf: canvas.Canvas, page_number: int) -> None:
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 7.5)
    pdf.drawString(34, 18, "Ellina moodboard corrections - V2 review pack")
    pdf.drawRightString(PAGE_W - 34, 18, f"Page {page_number}")


def cover_page(pdf: canvas.Canvas, root: Path) -> None:
    pdf.setFillColor(CREAM)
    pdf.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    pdf.setFillColor(PINK)
    pdf.circle(PAGE_W - 105, PAGE_H - 80, 115, fill=1, stroke=0)
    pdf.setFillColor(ACCENT)
    pdf.roundRect(52, PAGE_H - 144, 92, 24, 12, fill=1, stroke=0)
    pdf.setFillColor(PAPER)
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawCentredString(98, PAGE_H - 136, "REVIEW PACK")

    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 30)
    pdf.drawString(52, PAGE_H - 205, "Ellina moodboard corrections")
    pdf.setFont("Helvetica", 21)
    pdf.drawString(52, PAGE_H - 239, "V2 pilot - before, feedback and updated fix")

    pdf.setFillColor(PAPER)
    pdf.roundRect(52, 118, 515, 170, 14, fill=1, stroke=0)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(72, 258, "What is in this pack")
    bullets = [
        "Four representative fixes from Ellina's 360-action review manifest.",
        "The original V1 moodboard beside the implemented V2 asset.",
        "Ellina's feedback, a precise change log and the self-review decision.",
        "A reversible iOS implementation: no original moodboard was overwritten or deleted.",
    ]
    draw_bullets(pdf, bullets, 72, 232, 465)

    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(610, 246, "IMPLEMENTED")
    pdf.setFillColor(GREEN)
    pdf.setFont("Helvetica-Bold", 42)
    pdf.drawString(610, 196, "4")
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(610, 178, "V2 moodboards")
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(610, 142, "ROLLBACK")
    pdf.setFillColor(GREEN)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(610, 116, "V1 preserved")
    footer(pdf, 1)
    pdf.showPage()


def comparison_page(pdf: canvas.Canvas, root: Path, pilot: dict, page_number: int) -> None:
    pdf.setFillColor(CREAM)
    pdf.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 23)
    pdf.drawString(34, PAGE_H - 46, pilot["brand"])
    pdf.setFillColor(ACCENT)
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawRightString(PAGE_W - 34, PAGE_H - 42, f'{pilot["review_id"]}  /  {pilot["category"].upper()}')

    pdf.setFillColor(PAPER)
    pdf.roundRect(34, PAGE_H - 116, PAGE_W - 68, 50, 10, fill=1, stroke=0)
    pdf.setFillColor(ACCENT)
    pdf.setFont("Helvetica-Bold", 8.5)
    pdf.drawString(48, PAGE_H - 85, "ELLINA'S FEEDBACK")
    draw_wrapped(pdf, pilot["feedback"], 148, PAGE_H - 85, PAGE_W - 200, size=9, leading=11, max_lines=3)

    left_x, right_x = 54, 474
    image_y, image_w, image_h = 139, 236, 320
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawCentredString(left_x + image_w / 2, 473, "ORIGINAL - V1")
    pdf.drawCentredString(right_x + image_w / 2, 473, "UPDATED - V2")
    draw_fitted_image(pdf, root / pilot["original"], left_x, image_y, image_w, image_h)
    draw_fitted_image(pdf, root / pilot["updated"], right_x, image_y, image_w, image_h)

    pdf.setFillColor(PINK)
    pdf.circle(PAGE_W / 2, 316, 35, fill=1, stroke=0)
    pdf.setFillColor(ACCENT)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawCentredString(PAGE_W / 2, 320, "V2")
    pdf.setFont("Helvetica", 8)
    pdf.drawCentredString(PAGE_W / 2, 306, "FIX")

    pdf.setFillColor(PAPER)
    pdf.roundRect(34, 42, 382, 78, 10, fill=1, stroke=0)
    pdf.roundRect(426, 42, 382, 78, 10, fill=1, stroke=0)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawString(48, 101, "WHAT CHANGED")
    draw_bullets(pdf, pilot["changes"], 48, 84, 350)
    pdf.setFont("Helvetica-Bold", 9)
    pdf.setFillColor(INK)
    pdf.drawString(440, 101, "SELF-REVIEW")
    pdf.setFillColor(GREEN)
    pdf.roundRect(722, 94, 68, 16, 8, fill=1, stroke=0)
    pdf.setFillColor(PAPER)
    pdf.setFont("Helvetica-Bold", 7.5)
    pdf.drawCentredString(756, 99, "APPROVED")
    draw_wrapped(pdf, pilot["review"], 440, 82, 348, size=8.7, leading=12)
    footer(pdf, page_number)
    pdf.showPage()


def summary_page(pdf: canvas.Canvas, page_number: int) -> None:
    pdf.setFillColor(CREAM)
    pdf.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 26)
    pdf.drawString(44, PAGE_H - 58, "Implementation, rollback and next batches")
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(44, PAGE_H - 79, "This pilot is a reviewed first batch, not a claim that all 360 actions are complete.")

    cards = [
        (
            44,
            314,
            "IMPLEMENTED NOW",
            [
                "4 approved moodboards installed as additive V2 assets.",
                "App lookup prefers a matching V2 file, then falls back to V1.",
                "Automated size, aspect and repeated-region screens passed.",
                "Human visual review approved all four final candidates.",
            ],
        ),
        (
            429,
            314,
            "SAFE ROLLBACK",
            [
                "Original V1 files remain byte-for-byte untouched.",
                "Each V2 asset has a manifest entry and source checksum.",
                "Removing or disabling one V2 lookup restores its V1 board.",
                "Superseded candidates remain available as review history.",
            ],
        ),
        (
            44,
            112,
            "NEXT REVIEW BATCHES",
            [
                "Prioritise the remaining P0 catalog and category decisions.",
                "Then regenerate P1 full-redo boards in small reviewable batches.",
                "Apply targeted P2 fixes after the brand-level direction is stable.",
                "Issue a new before-and-after PDF for each approved batch.",
            ],
        ),
        (
            429,
            112,
            "MANIFEST STATUS",
            [
                "360 total actions captured from Ellina's review.",
                "274 actions refer to existing boards; 86 request additions.",
                "43 catalog decisions require verification before generation.",
                "4 actions are implemented in this V2 pilot; 356 remain queued.",
            ],
        ),
    ]
    for x, y, title, bullets in cards:
        pdf.setFillColor(PAPER)
        pdf.roundRect(x, y, 365, 170, 12, fill=1, stroke=0)
        pdf.setFillColor(ACCENT)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(x + 18, y + 142, title)
        draw_bullets(pdf, bullets, x + 18, y + 116, 330)
    footer(pdf, page_number)
    pdf.showPage()


def build(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output), pagesize=landscape(A4), pageCompression=1)
    pdf.setTitle("Ellina Moodboard Corrections - V2 Review Pack")
    pdf.setAuthor("Brand Vibe project")
    cover_page(pdf, root)
    for page_number, pilot in enumerate(PILOTS, start=2):
        comparison_page(pdf, root, pilot, page_number)
    summary_page(pdf, len(PILOTS) + 2)
    pdf.save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/pdf/ellina_moodboard_v2_review.pdf"),
    )
    args = parser.parse_args()
    build(args.root.resolve(), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
