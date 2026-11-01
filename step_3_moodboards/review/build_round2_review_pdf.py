#!/usr/bin/env python3
"""Build the concise Ellina re-review PDF for non-green Round 2 items."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


PAGE_W, PAGE_H = landscape(A4)
INK = HexColor("#162036")
MUTED = HexColor("#667085")
PAPER = HexColor("#FFFEFB")
CREAM = HexColor("#F6F1E8")
PINK = HexColor("#EAC1C8")
ACCENT = HexColor("#9F3654")
GREEN = HexColor("#34735A")
LINE = HexColor("#D8D3CA")


def wrap(text: str, width: float, font: str, size: float) -> list[str]:
    words = str(text).split()
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


def preview(source: Path, cache_dir: Path) -> Path:
    key = hashlib.sha1(f"round2:{source.resolve()}:{source.stat().st_mtime_ns}".encode()).hexdigest()
    target = cache_dir / f"{key}.jpg"
    if target.exists():
        return target
    cache_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((900, 1200), Image.Resampling.LANCZOS)
        image.save(target, "JPEG", quality=80, optimize=True, progressive=True)
    return target


def panel(pdf: canvas.Canvas, source: Path, cache_dir: Path, x: float, y: float, w: float, h: float, label: str) -> None:
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 8.2)
    pdf.drawCentredString(x + w / 2, y + h + 10, label)
    pdf.setFillColor(PAPER)
    pdf.roundRect(x, y, w, h, 8, fill=1, stroke=0)
    image_path = preview(source, cache_dir)
    with Image.open(image_path) as opened:
        iw, ih = opened.size
    scale = min((w - 10) / iw, (h - 10) / ih)
    dw, dh = iw * scale, ih * scale
    pdf.drawImage(
        str(image_path),
        x + (w - dw) / 2,
        y + (h - dh) / 2,
        width=dw,
        height=dh,
        preserveAspectRatio=True,
        mask="auto",
    )
    pdf.setStrokeColor(LINE)
    pdf.roundRect(x, y, w, h, 8, fill=0, stroke=1)


def build(root: Path, manifest_path: Path, output: Path) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload["records"]
    cache_dir = root / "tmp" / "pdfs" / "round2_preview_cache"
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output), pagesize=landscape(A4), pageCompression=1)
    pdf.setTitle("Ellina Moodboard Re-review - Round 2")
    pdf.setAuthor("Brand Vibe Moodboard Review")

    margin = 32
    gap = 12
    panel_w = (PAGE_W - 2 * margin - 2 * gap) / 3
    panel_y = 55
    panel_h = 390

    for page_number, record in enumerate(records, start=1):
        pdf.setFillColor(CREAM)
        pdf.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        title = record["brand_name"]
        if record.get("variant"):
            title = f"{title} — {record['variant']}"
        pdf.setFillColor(INK)
        pdf.setFont("Helvetica-Bold", 20)
        pdf.drawString(margin, PAGE_H - 42, title)
        pdf.setFillColor(MUTED)
        pdf.setFont("Helvetica", 8)
        pdf.drawRightString(PAGE_W - margin, PAGE_H - 39, f"{record['review_id']} · {record['category'].upper()} · SOURCE PAGE {record['source_page']}")

        pdf.setFillColor(PINK)
        pdf.roundRect(margin, PAGE_H - 100, PAGE_W - 2 * margin, 39, 8, fill=1, stroke=0)
        pdf.setFillColor(ACCENT)
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawString(margin + 12, PAGE_H - 76, "NEW ROUND 2 COMMENT")
        lines = wrap(record["round2_feedback"], PAGE_W - 2 * margin - 170, "Helvetica-Bold", 11)
        pdf.setFillColor(INK)
        pdf.setFont("Helvetica-Bold", 11)
        text_x = margin + 150
        text_y = PAGE_H - 77
        for line in lines[:2]:
            pdf.drawString(text_x, text_y, line)
            text_y -= 12

        sources = [
            root / record["round1_path"],
            root / record["updated_path"],
            root / record["reference_path"],
        ]
        missing = [str(path) for path in sources if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing PDF inputs: " + ", ".join(missing))
        labels = ["ROUND 1 UPDATED", "ROUND 2 UPDATED", "GOOGLE IMAGES REFERENCE"]
        for index, (source, label) in enumerate(zip(sources, labels)):
            panel(pdf, source, cache_dir, margin + index * (panel_w + gap), panel_y, panel_w, panel_h, label)

        pdf.setFillColor(GREEN)
        pdf.roundRect(margin, 23, 96, 17, 8.5, fill=1, stroke=0)
        pdf.setFillColor(PAPER)
        pdf.setFont("Helvetica-Bold", 7)
        pdf.drawCentredString(margin + 48, 28.2, "UPDATED FOR RE-REVIEW")
        pdf.setFillColor(MUTED)
        pdf.setFont("Helvetica", 7.5)
        pdf.drawString(margin + 108, 28, "Only the latest canonical board is promoted to the app repositories.")
        pdf.drawRightString(PAGE_W - margin, 28, f"Page {page_number} of {len(records)}")
        pdf.showPage()

    pdf.save()
    print(f"Wrote {len(records)} pages to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="step_3_moodboards/review/round2_re_review_manifest.json")
    parser.add_argument("--output", default="output/pdf/ellina_moodboards_round2_review.pdf")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    build(root, root / args.manifest, root / args.output)


if __name__ == "__main__":
    main()
