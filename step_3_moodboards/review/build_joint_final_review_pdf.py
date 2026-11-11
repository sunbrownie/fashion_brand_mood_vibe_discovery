#!/usr/bin/env python3
"""Merge the latest correction reviews and category cluster atlas into one final deck."""

from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROUND2 = ROOT / "output" / "pdf" / "ellina_moodboards_round2_review.pdf"
DEFAULT_SWIM = ROOT / "output" / "pdf" / "swimwear_lingerie_review_round.pdf"
DEFAULT_CLUSTERS = ROOT / "output" / "pdf" / "all_category_cluster_review.pdf"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "ellina_final_joint_review.pdf"

PAGE_W, PAGE_H = landscape(A4)
INK = HexColor("#152238")
PAPER = HexColor("#F6F1E8")
PINK = HexColor("#ED7F95")
GREEN = HexColor("#82A68B")
WHITE = HexColor("#FFFEFB")
MUTED = HexColor("#647083")


def build_cover(path: Path, counts: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path), pagesize=(PAGE_W, PAGE_H), invariant=1, pageCompression=1)
    pdf.setTitle("ModeNova final Ellina review")
    pdf.setAuthor("ModeNova")
    pdf.setFillColor(INK)
    pdf.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    pdf.setFillColor(PINK)
    pdf.circle(PAGE_W - 92, PAGE_H - 92, 58, fill=1, stroke=0)
    pdf.setFillColor(GREEN)
    pdf.circle(PAGE_W - 158, PAGE_H - 148, 34, fill=1, stroke=0)
    pdf.setFillColor(WHITE)
    pdf.setFont("Helvetica-Bold", 31)
    pdf.drawString(52, PAGE_H - 110, "Final moodboard and cluster review")
    pdf.setFont("Helvetica", 13)
    pdf.drawString(54, PAGE_H - 143, "Latest boards only - new Ellina decisions, swimwear/lingerie and public archetypes")
    pdf.setFillColor(PAPER)
    pdf.setFont("Helvetica-Bold", 12)
    labels = (
        ("1", "Three latest Ellina decisions", counts[0]),
        ("2", "Swimwear & lingerie corrections", counts[1]),
        ("3", "Category-grouped cluster atlas", counts[2]),
    )
    y = PAGE_H - 230
    for number, label, pages in labels:
        pdf.setFillColor(PINK if number != "3" else GREEN)
        pdf.circle(68, y + 3, 14, fill=1, stroke=0)
        pdf.setFillColor(WHITE)
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawCentredString(68, y - 1, number)
        pdf.setFillColor(WHITE)
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(96, y, label)
        pdf.setFillColor(MUTED)
        pdf.setFont("Helvetica", 10)
        pdf.drawString(380, y, f"{pages} pages")
        y -= 56
    pdf.setFillColor(PAPER)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(54, 48, "Ordered by category and public archetype - four large portrait boards in one row.")
    pdf.showPage()
    pdf.save()


def append_pdf(writer: PdfWriter, source: Path) -> int:
    reader = PdfReader(str(source))
    for page in reader.pages:
        writer.add_page(page)
    return len(reader.pages)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round2", type=Path, default=DEFAULT_ROUND2)
    parser.add_argument("--swim", type=Path, default=DEFAULT_SWIM)
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sources = [args.round2.resolve(), args.swim.resolve(), args.clusters.resolve()]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing joint review inputs: " + ", ".join(missing))

    counts = tuple(len(PdfReader(str(path)).pages) for path in sources)
    cover = ROOT / "tmp" / "pdfs" / "joint_final_review_cover.pdf"
    build_cover(cover, counts)

    writer = PdfWriter()
    append_pdf(writer, cover)
    starts = []
    for source in sources:
        starts.append(len(writer.pages))
        append_pdf(writer, source)
    writer.add_outline_item("Three latest Ellina decisions", starts[0])
    writer.add_outline_item("Swimwear and lingerie corrections", starts[1])
    writer.add_outline_item("Category-grouped cluster atlas", starts[2])
    writer.add_metadata({"/Title": "ModeNova final moodboard and cluster review", "/Author": "ModeNova"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        writer.write(handle)
    print(f"Wrote {len(writer.pages)} pages to {args.output}")


if __name__ == "__main__":
    main()
