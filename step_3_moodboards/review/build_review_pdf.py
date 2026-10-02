#!/usr/bin/env python3
"""Build the complete 360-fixture Ellina V2 review PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageOps
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
BLUE = HexColor("#456990")
LINE = HexColor("#D7D2C8")


def clean_text(value: object) -> str:
    text = str(value or "")
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2011": "-", "\u2026": "...",
        "\u00a0": " ", "\u200b": "",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r"\s+", " ", text).strip()


def normalized(value: str) -> str:
    folded = "".join(
        character
        for character in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(character)
    )
    return "_".join(re.findall(r"[a-z0-9]+", folded.lower().replace("&", " and ")))


ORIGINAL_ALIASES = {
    ("clothes", "gravel_and_gold"): "gravel_gold",
    ("clothes", "rag_and_bone"): "ragbone",
    ("jewellery", "rag_and_bone"): "ragbone",
    ("jewellery", "tod_s"): "tods",
    ("bags", "l_agence"): "lagence",
}


def original_moodboard(root: Path, record: dict) -> Path | None:
    folder = root / "moodboards_creation" / "moodboards" / record["category"]
    slug = normalized(record["brand_name"])
    stems = [slug]
    alias = ORIGINAL_ALIASES.get((record["category"], slug))
    if alias:
        stems.insert(0, alias)
    stems.extend([slug.replace("_and_", "_"), slug.replace("_", "")])
    for stem in dict.fromkeys(stems):
        for extension in (".jpg", ".jpeg", ".png", ".webp"):
            path = folder / f"{stem}{extension}"
            if path.exists():
                return path
    compact_slug = slug.replace("_", "")
    for path in folder.iterdir():
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        normalized_stem = normalized(path.stem)
        if normalized_stem == slug or normalized_stem.replace("_", "") == compact_slug:
            return path
    return None


def updated_moodboards(root: Path, record: dict) -> list[Path]:
    ios_root = root / "ios" / "BrandMoodboardFinder"
    return [ios_root / value for value in record.get("ios_v2_paths", [])]


def reference_sources(root: Path) -> dict[str, dict]:
    path = root / "step_3_moodboards" / "review" / "reference_sources.json"
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    return {record["review_id"]: record for record in records}


def wrap_lines(text: str, font: str, size: float, width: float) -> list[str]:
    words = clean_text(text).split()
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
    *,
    font: str = "Helvetica",
    size: float = 8.5,
    leading: float = 10.5,
    color=INK,
    max_lines: int | None = None,
) -> float:
    lines = wrap_lines(text, font, size, width)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        final = lines[-1]
        while final and stringWidth(final + "...", font, size) > width:
            final = final[:-1].rstrip()
        lines[-1] = final + "..."
    pdf.setFillColor(color)
    pdf.setFont(font, size)
    for line in lines:
        pdf.drawString(x, y, line)
        y -= leading
    return y


def cached_review_image(source: Path, cache_dir: Path, quality: int = 72) -> Path:
    key = hashlib.sha1(f"compact-v2:q{quality}:{source.resolve()}:{source.stat().st_mtime_ns}".encode()).hexdigest()
    target = cache_dir / f"{key}.jpg"
    if target.exists():
        return target
    cache_dir.mkdir(parents=True, exist_ok=True)
    with PILImage.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((760, 1064), PILImage.Resampling.LANCZOS)
        image.save(target, format="JPEG", quality=quality, optimize=True, progressive=True)
    return target


def draw_fitted_image(
    pdf: canvas.Canvas,
    source: Path,
    cache_dir: Path,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    quality: int = 72,
) -> None:
    path = cached_review_image(source, cache_dir, quality)
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


def draw_updated_panel(pdf: canvas.Canvas, paths: list[Path], cache_dir: Path, x: float, y: float, w: float, h: float) -> None:
    if len(paths) == 1:
        draw_fitted_image(pdf, paths[0], cache_dir, x, y, w, h)
        return
    gap = 6
    cell_w = (w - gap) / 2
    for index, path in enumerate(paths[:2]):
        cell_x = x + index * (cell_w + gap)
        draw_fitted_image(pdf, path, cache_dir, cell_x, y, cell_w, h)
        label = "PLAY" if "_play" in path.stem else "MAIN LINE / AVANT-GARDE"
        pdf.setFillColor(MUTED)
        pdf.setFont("Helvetica-Bold", 6.5)
        pdf.drawCentredString(cell_x + cell_w / 2, y + h - 16, label)


def footer(pdf: canvas.Canvas, page_number: int, total_pages: int) -> None:
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 7.5)
    pdf.drawString(34, 18, "Ellina review - complete V2 implementation pack")
    pdf.drawRightString(PAGE_W - 34, 18, f"Page {page_number} of {total_pages}")


def badge(pdf: canvas.Canvas, text: str, x: float, y: float, width: float, color=GREEN) -> None:
    pdf.setFillColor(color)
    pdf.roundRect(x, y, width, 17, 8.5, fill=1, stroke=0)
    pdf.setFillColor(PAPER)
    pdf.setFont("Helvetica-Bold", 7.2)
    pdf.drawCentredString(x + width / 2, y + 5.3, clean_text(text).upper())


def implementation_copy(record: dict) -> tuple[str, str, str]:
    action = record["recommended_action"]
    if action == "catalog_add_and_generate":
        title = "ADDED TO CATALOG + V2 GENERATED"
        detail = (
            f"Bundled local overlay added this {record['category']} brand with official-site, audience and price metadata. "
            f"Taste and recommendation fallback: {record.get('taste_proxy_brand_name', 'validated category proxy')}."
        )
        review = record.get("self_review_note", "Approved after visual review; bundled asset and catalog metadata validated.")
    elif record.get("implementation_status") == "implemented_v2":
        title = "APPROVED V2 REGENERATION"
        count = len(record.get("ios_v2_paths", []))
        detail = f"Implemented as {count} additive V2 asset{'s' if count != 1 else ''}. The app prefers V2 when present and retains V1 for rollback."
        review = record.get("self_review_note", "Approved after visual and asset validation.")
    else:
        resolved = record.get("resolved_action", "catalog decision")
        title = resolved.replace("_", " ").upper()
        detail = record.get("resolution_note") or "Accepted Ellina's category correction in the runtime catalog layer."
        review = "Resolved without generating a replacement board. The original image remains preserved and the decision is reversible."
    return clean_text(title), clean_text(detail), clean_text(review)


def draw_empty_original_panel(pdf: canvas.Canvas, x: float, y: float, w: float, h: float) -> None:
    pdf.setFillColor(PAPER)
    pdf.roundRect(x, y, w, h, 8, fill=1, stroke=0)
    pdf.setStrokeColor(LINE)
    pdf.roundRect(x, y, w, h, 8, fill=0, stroke=1)
    pdf.setFillColor(PINK)
    pdf.circle(x + w / 2, y + h / 2 + 28, 38, fill=1, stroke=0)
    pdf.setFillColor(ACCENT)
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawCentredString(x + w / 2, y + h / 2 + 22, "NEW")
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawCentredString(x + w / 2, y + h / 2 - 28, "NO V1 MOODBOARD")
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 8)
    pdf.drawCentredString(x + w / 2, y + h / 2 - 44, "Ellina requested a new catalog brand")


def draw_decision_panel(pdf: canvas.Canvas, title: str, detail: str, x: float, y: float, w: float, h: float) -> None:
    pdf.setFillColor(PAPER)
    pdf.roundRect(x, y, w, h, 8, fill=1, stroke=0)
    pdf.setStrokeColor(LINE)
    pdf.roundRect(x, y, w, h, 8, fill=0, stroke=1)
    color = BLUE if "MERGE" in title else ACCENT
    badge(pdf, "catalog decision", x + 22, y + h - 48, 104, color)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 17)
    for index, line in enumerate(wrap_lines(title, "Helvetica-Bold", 17, w - 44)[:3]):
        pdf.drawString(x + 22, y + h - 86 - index * 22, line)
    draw_wrapped(pdf, detail, x + 22, y + h - 166, w - 44, size=9, leading=13, max_lines=9)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica-Bold", 8)
    pdf.drawString(x + 22, y + 29, "NO NEW IMAGE REQUIRED")


def fixture_page(
    pdf: canvas.Canvas,
    root: Path,
    cache_dir: Path,
    record: dict,
    reference: dict,
    page_number: int,
    total_pages: int,
) -> None:
    original = None if record["request_type"] == "new_brand" else original_moodboard(root, record)
    updated = updated_moodboards(root, record)
    reference_image = root / reference["screenshot_path"]
    visual_change = record.get("implementation_status") == "implemented_v2"
    title, detail, _ = implementation_copy(record)

    pdf.setFillColor(CREAM)
    pdf.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawString(34, PAGE_H - 43, clean_text(record["brand_name"]))
    pdf.setFillColor(ACCENT)
    pdf.setFont("Helvetica-Bold", 8.5)
    source = f"{record['review_id']}  /  {record['category'].upper()}  /  {clean_text(record['source_pdf'])}"
    pdf.drawRightString(PAGE_W - 34, PAGE_H - 39, source)

    pdf.setFillColor(PAPER)
    pdf.roundRect(34, PAGE_H - 112, PAGE_W - 68, 50, 9, fill=1, stroke=0)
    pdf.setFillColor(ACCENT)
    pdf.setFont("Helvetica-Bold", 8)
    pdf.drawString(47, PAGE_H - 82, "ELLINA'S FEEDBACK")
    draw_wrapped(pdf, record["reviewer_note"], 146, PAGE_H - 82, PAGE_W - 193, size=8.6, leading=10.5, max_lines=3)

    left_x, middle_x, right_x = 30, 301, 572
    image_y, image_w, image_h = 70, 240, 385
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawCentredString(left_x + image_w / 2, 467, "ORIGINAL - V1" if original else "REQUEST - NO V1")
    pdf.drawCentredString(middle_x + image_w / 2, 467, "UPDATED - V2" if visual_change else "IMPLEMENTED CATALOG DECISION")
    pdf.drawCentredString(right_x + image_w / 2, 467, "GOOGLE IMAGES REFERENCE")

    if original:
        draw_fitted_image(pdf, original, cache_dir, left_x, image_y, image_w, image_h)
    else:
        draw_empty_original_panel(pdf, left_x, image_y, image_w, image_h)

    if visual_change:
        draw_updated_panel(pdf, updated, cache_dir, middle_x, image_y, image_w, image_h)
    else:
        draw_decision_panel(pdf, title, detail, middle_x, image_y, image_w, image_h)

    draw_fitted_image(pdf, reference_image, cache_dir, right_x, image_y, image_w, image_h, quality=45)
    official_url = clean_text(reference["official_url"])
    pdf.setFillColor(BLUE)
    pdf.setFont("Helvetica", 5.8)
    link_lines = wrap_lines(official_url, "Helvetica", 5.8, image_w)
    link_y = 56
    for line in link_lines[:2]:
        pdf.drawCentredString(right_x + image_w / 2, link_y, line)
        link_y -= 6.8
    pdf.linkURL(reference["official_url"], (right_x, 42, right_x + image_w, 62), relative=0)

    footer(pdf, page_number, total_pages)
    pdf.showPage()


def validate_inputs(root: Path, records: list[dict], references: dict[str, dict]) -> None:
    if len(records) != 360:
        raise RuntimeError(f"Expected 360 fixtures, found {len(records)}")
    missing_originals = [record["review_id"] for record in records if record["request_type"] != "new_brand" and original_moodboard(root, record) is None]
    missing_updates = [
        record["review_id"]
        for record in records
        if record.get("implementation_status") == "implemented_v2"
        and (not updated_moodboards(root, record) or not all(path.exists() for path in updated_moodboards(root, record)))
    ]
    unresolved = [record["review_id"] for record in records if not record["implementation_status"].startswith("implemented")]
    missing_references = [
        record["review_id"]
        for record in records
        if record["review_id"] not in references
        or not (root / references[record["review_id"]]["screenshot_path"]).exists()
    ]
    if missing_originals or missing_updates or unresolved or missing_references:
        raise RuntimeError(
            f"PDF inputs invalid: missing originals={missing_originals}, missing updates={missing_updates}, "
            f"unresolved={unresolved}, missing references={missing_references}"
        )


def build(root: Path, output: Path, cache_dir: Path) -> None:
    manifest_path = root / "step_3_moodboards" / "review" / "ellina_correction_manifest.json"
    records = json.loads(manifest_path.read_text(encoding="utf-8"))["records"]
    references = reference_sources(root)
    validate_inputs(root, records, references)
    output.parent.mkdir(parents=True, exist_ok=True)
    total_pages = len(records)
    pdf = canvas.Canvas(str(output), pagesize=landscape(A4), pageCompression=1)
    pdf.setTitle("Ellina Moodboard Corrections - Complete V2 Review")
    pdf.setAuthor("Brand Vibe project")
    pdf.setSubject("All 360 review fixtures with original feedback and implemented outcomes")
    last_group = None
    for index, record in enumerate(records, start=1):
        group = (record["source_pdf"], record["category"])
        if group != last_group:
            key = f"section-{record['review_id']}"
            pdf.bookmarkPage(key)
            pdf.addOutlineEntry(f"{clean_text(group[0])} - {group[1].title()}", key, level=0, closed=False)
            last_group = group
        fixture_page(pdf, root, cache_dir, record, references[record["review_id"]], index, total_pages)
    pdf.save()
    print(f"Created {output} with {total_pages} pages for {len(records)} fixtures.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path("output/pdf/ellina_complete_v2_review.pdf"))
    parser.add_argument("--cache-dir", type=Path, default=Path("tmp/pdfs/ellina_review_images"))
    args = parser.parse_args()
    build(args.root.resolve(), args.output.resolve(), args.cache_dir.resolve())


if __name__ == "__main__":
    main()
