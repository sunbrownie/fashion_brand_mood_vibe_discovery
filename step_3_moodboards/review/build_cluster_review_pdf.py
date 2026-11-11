#!/usr/bin/env python3
"""Build the deterministic all-category customer-facing archetype review PDF.

The report reads the current iOS assignment export, joins each brand to the
canonical moodboard recorded in ``brand_metadata.csv``, and groups brands by
category and customer-facing archetype.  Technical clusters remain available
for filtering and app logic but are deliberately not exposed as separate PDF
subgroups.  Every brand is rendered exactly once and missing images are
deliberately visible.

Examples
--------
Build the complete report::

    python step_3_moodboards/review/build_cluster_review_pdf.py

Build one cluster for layout validation::

    python step_3_moodboards/review/build_cluster_review_pdf.py \
      --cluster shoes:5 --output tmp/pdfs/cluster_review_sample.pdf
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageOps
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "recommender" / "hosting_bundle" / "data"
DEFAULT_ASSIGNMENTS = DATA / "ios_brand_cluster_assignments.csv"
DEFAULT_SUMMARY = DATA / "ios_cluster_summary.csv"
DEFAULT_METADATA = DATA / "brand_metadata.csv"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "all_category_cluster_review.pdf"
DEFAULT_CACHE = ROOT / "tmp" / "pdfs" / "cluster_review_thumbnails"

CATEGORY_ORDER = ("clothes", "shoes", "bags", "jewellery", "swimwear_lingerie")
CATEGORY_LABELS = {
    "clothes": "Clothes",
    "shoes": "Shoes",
    "bags": "Bags",
    "jewellery": "Jewellery",
    "swimwear_lingerie": "Swimwear & Lingerie",
}

PAGE_W, PAGE_H = landscape(A4)
MARGIN_X = 30
FOOTER_H = 26
HEADER_BOTTOM = 458
GRID_TOP = 445
GRID_BOTTOM = 39
GRID_COLS = 4
GRID_ROWS = 1
CARDS_PER_PAGE = GRID_COLS * GRID_ROWS
GAP_X = 9
GAP_Y = 9

INK = HexColor("#152238")
MUTED = HexColor("#647083")
PAPER = HexColor("#F6F1E8")
CREAM = HexColor("#EEE4D4")
WHITE = HexColor("#FFFEFB")
PINK = HexColor("#ED7F95")
GREEN = HexColor("#82A68B")
PALE_GREEN = HexColor("#E8F0E9")
PALE_PINK = HexColor("#FBE9ED")
LINE = HexColor("#D9D1C6")

PUBLIC_ARCHETYPE_ORDER = (
    "The Expedition",
    "The Understated",
    "The Disruptor",
    "The Dreamer",
    "The Underground",
    "Modern Minimalist",
    "After Midnight",
    "The Trove",
    "Sun Bleached",
    "The Workshop",
    "The Dresser",
    "Endorphin",
    "The Wanderer",
    "The Alumni",
    "Riviera",
)

ARCHETYPE_COLORS = {
    "The Expedition": HexColor("#4F766A"),
    "The Understated": HexColor("#7B756D"),
    "The Disruptor": HexColor("#A34958"),
    "The Dreamer": HexColor("#B27FA1"),
    "The Underground": HexColor("#4D5269"),
    "Modern Minimalist": HexColor("#728899"),
    "After Midnight": HexColor("#3E3C55"),
    "The Trove": HexColor("#B8734D"),
    "Sun Bleached": HexColor("#C69A63"),
    "The Workshop": HexColor("#7B624C"),
    "The Dresser": HexColor("#A76478"),
    "Endorphin": HexColor("#4A8E8A"),
    "The Wanderer": HexColor("#8A7654"),
    "The Alumni": HexColor("#526B83"),
    "Riviera": HexColor("#3E8FA0"),
}


@dataclass(frozen=True)
class ImageRef:
    path: Path | None
    source: str
    attempted: str


@dataclass(frozen=True)
class ClusterBlock:
    category: str
    cluster: int
    title: str
    top_terms: str
    supporting_terms: str
    dominant_public_name: str
    dominant_public_description: str
    rows: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class ClusterPage:
    block: ClusterBlock
    rows: tuple[dict[str, str], ...]
    part: int
    parts: int
    archetype_cluster_index: int
    archetype_cluster_count: int


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def key_text(value: object) -> str:
    return " ".join(clean(value).casefold().split())


def cluster_number(value: object) -> int:
    text = clean(value)
    if not text:
        raise ValueError("Blank technical cluster ID")
    return int(float(text))


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.replace("&", " and "))
    ascii_value = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "_", ascii_value.lower()).strip("_")


def stable_brand_sort(row: dict[str, str]) -> tuple[str, str]:
    name = clean(row.get("brand_name"))
    return (key_text(name), name)


def modal(rows: Sequence[dict[str, str]], field: str) -> str:
    values = [clean(row.get(field)) for row in rows if clean(row.get(field))]
    if not values:
        return ""
    counts = Counter(values)
    return sorted(counts, key=lambda item: (-counts[item], item.casefold(), item))[0]


def archetype_sort_key(name: str) -> tuple[int, str]:
    try:
        return (PUBLIC_ARCHETYPE_ORDER.index(name), name.casefold())
    except ValueError:
        return (len(PUBLIC_ARCHETYPE_ORDER), name.casefold())


def archetype_color(name: str):
    return ARCHETYPE_COLORS.get(name, GREEN)


def count_label(count: int, singular: str) -> str:
    return f"{count} {singular if count == 1 else singular + 's'}"


def metadata_index(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (key_text(row.get("category")), key_text(row.get("brand_name")))
        if key[0] and key[1] and key not in result:
            result[key] = row
    return result


def summary_index(rows: Iterable[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    result: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        category = key_text(row.get("category"))
        if category and clean(row.get("cluster")):
            result[(category, cluster_number(row["cluster"]))] = row
    return result


def build_blocks(
    assignments: list[dict[str, str]],
    summaries: dict[tuple[str, int], dict[str, str]],
) -> list[ClusterBlock]:
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    seen_brand_keys: set[tuple[str, int, str]] = set()
    for row in assignments:
        category = key_text(row.get("category"))
        cluster = cluster_number(row.get("cluster"))
        brand = key_text(row.get("brand_name"))
        key = (category, cluster, brand)
        if key in seen_brand_keys:
            raise ValueError(f"Duplicate assignment row: {category} cluster {cluster}, {brand}")
        seen_brand_keys.add(key)
        public_name = clean(row.get("public_display_name")) or clean(row.get("public_cluster_name"))
        if not public_name:
            summary = summaries.get((category, cluster), {})
            public_name = clean(summary.get("dominant_public_display_name")) or clean(
                summary.get("dominant_public_cluster_name")
            ) or "Unmapped public archetype"
        grouped[(category, public_name, cluster)].append(row)

    order = {category: index for index, category in enumerate(CATEGORY_ORDER)}
    blocks: list[ClusterBlock] = []
    for category, public_name, cluster in sorted(
        grouped,
        key=lambda key: (
            order.get(key[0], len(order)),
            key[0],
            archetype_sort_key(key[1]),
            key[2],
        ),
    ):
        rows = tuple(sorted(grouped[(category, public_name, cluster)], key=stable_brand_sort))
        summary = summaries.get((category, cluster), {})
        public_description = modal(rows, "public_display_description") or modal(
            rows, "public_cluster_description"
        )
        if not public_description:
            public_description = clean(summary.get("dominant_public_display_description")) or clean(
                summary.get("dominant_public_cluster_description")
            )
        blocks.append(
            ClusterBlock(
                category=category,
                cluster=cluster,
                title=clean(summary.get("cluster_title")) or modal(rows, "cluster_title") or "Untitled",
                top_terms=clean(summary.get("top_aesthetic_terms"))
                or modal(rows, "top_aesthetic_terms"),
                supporting_terms=clean(summary.get("supporting_terms"))
                or modal(rows, "supporting_terms"),
                dominant_public_name=public_name,
                dominant_public_description=public_description or "No public description is currently assigned.",
                rows=rows,
            )
        )
    return blocks


def parse_cluster_spec(value: str) -> tuple[str, int]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Use CATEGORY:CLUSTER, for example shoes:5")
    category, cluster = value.rsplit(":", 1)
    try:
        return key_text(category), int(cluster)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid cluster spec: {value}") from exc


def select_blocks(
    blocks: list[ClusterBlock],
    categories: Sequence[str],
    clusters: Sequence[tuple[str, int]],
    max_clusters: int | None,
) -> list[ClusterBlock]:
    category_filter = {key_text(item) for item in categories}
    cluster_filter = set(clusters)
    selected = [
        block
        for block in blocks
        if (not category_filter or block.category in category_filter)
        and (not cluster_filter or (block.category, block.cluster) in cluster_filter)
    ]
    if max_clusters is not None:
        selected = selected[:max_clusters]
    if not selected:
        raise ValueError("No technical clusters matched the requested filters")
    return selected


def merge_public_archetype_blocks(blocks: Sequence[ClusterBlock]) -> list[ClusterBlock]:
    """Combine technical-source fragments into one visible public archetype."""
    grouped: dict[tuple[str, str], list[ClusterBlock]] = defaultdict(list)
    for block in blocks:
        grouped[(block.category, block.dominant_public_name)].append(block)
    order = {category: index for index, category in enumerate(CATEGORY_ORDER)}
    merged: list[ClusterBlock] = []
    for (category, public_name), source_blocks in sorted(
        grouped.items(),
        key=lambda item: (
            order.get(item[0][0], len(order)),
            item[0][0],
            archetype_sort_key(item[0][1]),
        ),
    ):
        rows = tuple(
            sorted(
                (row for block in source_blocks for row in block.rows),
                key=stable_brand_sort,
            )
        )
        merged.append(
            ClusterBlock(
                category=category,
                cluster=-1,
                title=public_name,
                top_terms="",
                supporting_terms="",
                dominant_public_name=public_name,
                dominant_public_description=(
                    modal(rows, "public_display_description")
                    or modal(rows, "public_cluster_description")
                    or "No public description is currently assigned."
                ),
                rows=rows,
            )
        )
    return merged


def resolve_moodboard(
    row: dict[str, str],
    metadata: dict[tuple[str, str], dict[str, str]],
) -> ImageRef:
    category = key_text(row.get("category"))
    brand_name = clean(row.get("brand_name"))
    meta = metadata.get((category, key_text(brand_name)), {})
    explicit = clean(meta.get("moodboard_path"))
    meta_slug = clean(meta.get("slug"))
    slugs = list(dict.fromkeys(value for value in (meta_slug, slugify(brand_name)) if value))

    candidates: list[tuple[Path, str]] = []
    if explicit:
        explicit_path = Path(explicit)
        candidates.append(
            (explicit_path if explicit_path.is_absolute() else DATA / explicit_path, "metadata")
        )
    for slug in slugs:
        for extension in (".jpg", ".jpeg", ".png", ".webp"):
            candidates.append((DATA / "moodboards" / category / f"{slug}{extension}", "canonical-slug"))
    for slug in slugs:
        candidates.append(
            (
                ROOT
                / "moodboards_creation"
                / "moodboards_corrected"
                / "v2"
                / category
                / f"{slug}.png",
                "approved-v2",
            )
        )
        for extension in (".jpg", ".png"):
            candidates.append(
                (
                    ROOT
                    / "ios"
                    / "BrandMoodboardFinder"
                    / "BrandMoodboardFinder"
                    / "BrandMoodboards"
                    / f"v2__{category}__{slug}{extension}",
                    "ios-v2",
                )
            )

    for path, source in candidates:
        if path.is_file():
            return ImageRef(path=path, source=source, attempted=explicit or slugs[0] if slugs else brand_name)
    attempted = explicit or (f"moodboards/{category}/{slugs[0]}" if slugs else brand_name)
    return ImageRef(path=None, source="missing", attempted=attempted)


def wrap_lines(text: str, font: str, size: float, width: float, max_lines: int) -> list[str]:
    words = clean(text).split()
    if not words:
        return []
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if not current or stringWidth(trial, font, size) <= width:
            current = trial
            continue
        lines.append(current)
        current = word
        if len(lines) == max_lines:
            break
    if len(lines) < max_lines and current:
        lines.append(current)
    truncated = len(lines) == max_lines and " ".join(lines) != " ".join(words)
    if truncated:
        while lines[-1] and stringWidth(lines[-1] + "…", font, size) > width:
            lines[-1] = lines[-1][:-1].rstrip()
        lines[-1] += "…"
    return lines


def draw_wrapped(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    font: str,
    size: float,
    leading: float,
    max_lines: int,
    color=INK,
) -> float:
    pdf.setFillColor(color)
    pdf.setFont(font, size)
    for line in wrap_lines(text, font, size, width, max_lines):
        pdf.drawString(x, y, line)
        y -= leading
    return y


def thumbnail(path: Path, cache_dir: Path, box: tuple[int, int] = (300, 450)) -> Path:
    stat = path.stat()
    token = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{box[0]}x{box[1]}"
    target = cache_dir / f"{hashlib.sha256(token.encode()).hexdigest()}.jpg"
    if target.is_file():
        return target
    cache_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail(box, Image.Resampling.LANCZOS)
        background = Image.new("RGB", box, (255, 254, 251))
        offset = ((box[0] - image.width) // 2, (box[1] - image.height) // 2)
        background.paste(image, offset)
        background.save(target, "JPEG", quality=76, optimize=True, progressive=False)
    return target


def draw_footer(pdf: canvas.Canvas, page_number: int, total_pages: int, snapshot: str) -> None:
    pdf.setStrokeColor(LINE)
    pdf.setLineWidth(0.5)
    pdf.line(MARGIN_X, FOOTER_H + 3, PAGE_W - MARGIN_X, FOOTER_H + 3)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 7.4)
    pdf.drawString(MARGIN_X, 14, f"ModeNova cluster review  •  data {snapshot}")
    pdf.drawRightString(PAGE_W - MARGIN_X, 14, f"Page {page_number} of {total_pages}")


def draw_cover(
    pdf: canvas.Canvas,
    blocks: Sequence[ClusterBlock],
    total_pages: int,
    snapshot: str,
) -> None:
    pdf.setFillColor(INK)
    pdf.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    pdf.setFillColor(PINK)
    pdf.circle(PAGE_W - 90, PAGE_H - 85, 55, fill=1, stroke=0)
    pdf.setFillColor(GREEN)
    pdf.circle(PAGE_W - 150, PAGE_H - 140, 32, fill=1, stroke=0)
    pdf.setFillColor(WHITE)
    pdf.setFont("Helvetica-Bold", 31)
    pdf.drawString(52, PAGE_H - 120, "All-category cluster review")
    pdf.setFont("Helvetica", 13)
    pdf.drawString(54, PAGE_H - 151, "Customer-facing archetypes and latest canonical moodboards")
    pdf.setFillColor(CREAM)
    pdf.setFont("Helvetica", 10)
    archetype_count = len({(block.category, block.dominant_public_name) for block in blocks})
    pdf.drawString(54, PAGE_H - 194, f"{archetype_count} category archetype groups  •  {sum(len(b.rows) for b in blocks)} brands")
    pdf.drawString(54, PAGE_H - 213, f"{len({b.category for b in blocks})} categories  •  {total_pages} pages")
    pdf.drawString(54, PAGE_H - 232, "Four large portrait moodboards in one row per review page")
    pdf.setFillColor(WHITE)
    pdf.setFont("Helvetica-Bold", 9)
    y = PAGE_H - 285
    for category in CATEGORY_ORDER:
        category_blocks = [block for block in blocks if block.category == category]
        if not category_blocks:
            continue
        label = CATEGORY_LABELS.get(category, category.replace("_", " ").title())
        pdf.drawString(55, y, label)
        pdf.setFont("Helvetica", 9)
        public_names = {block.dominant_public_name for block in category_blocks}
        pdf.drawString(205, y, f"{len(public_names)} archetypes  •  {sum(len(b.rows) for b in category_blocks)} brands")
        pdf.setFont("Helvetica-Bold", 9)
        y -= 25
    pdf.setFillColor(CREAM)
    pdf.setFont("Helvetica", 7.5)
    pdf.drawString(54, 48, f"Deterministic snapshot {snapshot}  •  missing images are shown explicitly")


def draw_category_divider(pdf: canvas.Canvas, category: str, blocks: Sequence[ClusterBlock]) -> None:
    label = CATEGORY_LABELS.get(category, category.replace("_", " ").title())
    pdf.setFillColor(PAPER)
    pdf.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    pdf.setFillColor(GREEN)
    pdf.roundRect(52, PAGE_H - 170, 86, 9, 4.5, fill=1, stroke=0)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 34)
    pdf.drawString(52, PAGE_H - 136, label)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 12)
    pdf.drawString(
        54,
        PAGE_H - 200,
        f"{len({block.dominant_public_name for block in blocks})} public archetypes  •  {sum(len(block.rows) for block in blocks)} assigned brands",
    )
    archetype_groups = []
    for name in sorted({block.dominant_public_name for block in blocks}, key=archetype_sort_key):
        grouped_blocks = [block for block in blocks if block.dominant_public_name == name]
        archetype_groups.append((name, grouped_blocks))
    split = math.ceil(len(archetype_groups) / 2)
    column_w = (PAGE_W - 126) / 2
    for index, (name, grouped_blocks) in enumerate(archetype_groups):
        column = index // split
        row = index % split
        x = 57 + column * column_w
        y = PAGE_H - 260 - row * 29
        pdf.setFillColor(archetype_color(name))
        pdf.roundRect(x, y - 4, 10, 10, 5, fill=1, stroke=0)
        pdf.setFillColor(INK)
        pdf.setFont("Helvetica-Bold", 8.3)
        line = wrap_lines(name, "Helvetica-Bold", 8.3, column_w - 92, 1)[0]
        pdf.drawString(x + 18, y, line)
        pdf.setFillColor(MUTED)
        pdf.setFont("Helvetica", 7.2)
        brand_count = sum(len(block.rows) for block in grouped_blocks)
        pdf.drawRightString(
            x + column_w - 12,
            y,
            count_label(brand_count, "brand"),
        )


def draw_archetype_divider(
    pdf: canvas.Canvas,
    category: str,
    public_name: str,
    blocks: Sequence[ClusterBlock],
) -> None:
    label = CATEGORY_LABELS.get(category, category.replace("_", " ").title())
    accent = archetype_color(public_name)
    description = blocks[0].dominant_public_description if blocks else ""
    pdf.setFillColor(PAPER)
    pdf.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    pdf.setFillColor(accent)
    pdf.rect(0, 0, 22, PAGE_H, fill=1, stroke=0)
    pdf.circle(PAGE_W - 104, PAGE_H - 100, 58, fill=1, stroke=0)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawString(54, PAGE_H - 78, label.upper())
    pdf.setFont("Helvetica-Bold", 34)
    pdf.drawString(52, PAGE_H - 137, public_name)
    draw_wrapped(
        pdf,
        description,
        54,
        PAGE_H - 190,
        550,
        "Helvetica",
        13,
        18,
        6,
        INK,
    )
    pdf.setFillColor(accent)
    pdf.roundRect(54, 83, 300, 34, 17, fill=1, stroke=0)
    pdf.setFillColor(WHITE)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawCentredString(
        204,
        95,
        count_label(sum(len(block.rows) for block in blocks), "brand"),
    )


def draw_cluster_header(pdf: canvas.Canvas, page: ClusterPage) -> None:
    block = page.block
    label = CATEGORY_LABELS.get(block.category, block.category.replace("_", " ").title())
    accent = archetype_color(block.dominant_public_name)
    pdf.setFillColor(PAPER)
    pdf.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    pdf.setFillColor(accent)
    pdf.rect(0, PAGE_H - 103, PAGE_W, 103, fill=1, stroke=0)
    pdf.setFillColor(WHITE)
    pdf.setFont("Helvetica-Bold", 8.2)
    pdf.drawString(MARGIN_X, PAGE_H - 29, label.upper())
    pdf.setFont("Helvetica-Bold", 24)
    pdf.drawString(MARGIN_X, PAGE_H - 63, block.dominant_public_name)
    pdf.setFont("Helvetica", 8.2)
    pdf.drawString(
        MARGIN_X,
        PAGE_H - 84,
        f"Latest canonical moodboards  •  page {page.part} of {page.parts}",
    )
    pdf.drawRightString(
        PAGE_W - MARGIN_X,
        PAGE_H - 84,
        count_label(len(block.rows), "brand"),
    )


def draw_missing(pdf: canvas.Canvas, x: float, y: float, width: float, height: float, attempted: str) -> None:
    pdf.setFillColor(PALE_PINK)
    pdf.roundRect(x, y, width, height, 5, fill=1, stroke=0)
    pdf.setStrokeColor(PINK)
    pdf.setLineWidth(1.2)
    cx, cy = x + width / 2, y + height / 2 + 8
    pdf.line(cx - 10, cy - 10, cx + 10, cy + 10)
    pdf.line(cx - 10, cy + 10, cx + 10, cy - 10)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 7)
    pdf.drawCentredString(cx, cy - 28, "MOODBOARD NOT FOUND")
    attempted_lines = wrap_lines(attempted, "Helvetica", 5.6, width - 14, 2)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 5.6)
    for index, line in enumerate(attempted_lines):
        pdf.drawCentredString(cx, cy - 39 - 7 * index, line)


def draw_brand_card(
    pdf: canvas.Canvas,
    row: dict[str, str],
    image_ref: ImageRef,
    cache_dir: Path,
    x: float,
    y: float,
    width: float,
    height: float,
) -> bool:
    pdf.setFillColor(WHITE)
    pdf.roundRect(x, y, width, height, 7, fill=1, stroke=0)
    image_x, image_y = x + 6, y + 30
    image_w, image_h = width - 12, height - 36
    missing = image_ref.path is None
    if missing:
        draw_missing(pdf, image_x, image_y, image_w, image_h, image_ref.attempted)
    else:
        try:
            thumb = thumbnail(image_ref.path, cache_dir)
            pdf.drawImage(
                ImageReader(str(thumb)),
                image_x,
                image_y,
                width=image_w,
                height=image_h,
                preserveAspectRatio=True,
                anchor="c",
                mask="auto",
            )
        except Exception:
            missing = True
            draw_missing(pdf, image_x, image_y, image_w, image_h, image_ref.path.name)

    brand = clean(row.get("brand_name")) or "Unnamed brand"
    draw_wrapped(pdf, brand, x + 7, y + 18, width - 14, "Helvetica-Bold", 8.3, 9.2, 2, INK)
    return missing


def snapshot_id(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:10]


def build_pdf(
    output: Path,
    assignments_path: Path,
    summary_path: Path,
    metadata_path: Path,
    cache_dir: Path,
    categories: Sequence[str],
    clusters: Sequence[tuple[str, int]],
    max_clusters: int | None,
) -> dict[str, int | str]:
    assignments = read_csv(assignments_path)
    summaries = summary_index(read_csv(summary_path))
    metadata = metadata_index(read_csv(metadata_path))
    technical_blocks = select_blocks(
        build_blocks(assignments, summaries), categories, clusters, max_clusters
    )
    blocks = merge_public_archetype_blocks(technical_blocks)

    selected_categories = [category for category in CATEGORY_ORDER if any(b.category == category for b in blocks)]
    selected_categories.extend(
        sorted({b.category for b in blocks} - set(selected_categories))
    )
    page_specs: list[tuple[str, object]] = [("cover", None)]
    for category in selected_categories:
        category_blocks = [block for block in blocks if block.category == category]
        page_specs.append(("category", category))
        public_names = sorted(
            {block.dominant_public_name for block in category_blocks},
            key=archetype_sort_key,
        )
        for public_name in public_names:
            archetype_blocks = [
                block for block in category_blocks if block.dominant_public_name == public_name
            ]
            page_specs.append(("archetype", (category, public_name, archetype_blocks)))
            for cluster_index, block in enumerate(archetype_blocks, start=1):
                parts = math.ceil(len(block.rows) / CARDS_PER_PAGE)
                for part in range(parts):
                    start = part * CARDS_PER_PAGE
                    page_specs.append(
                        (
                            "cluster",
                            ClusterPage(
                                block=block,
                                rows=block.rows[start : start + CARDS_PER_PAGE],
                                part=part + 1,
                                parts=parts,
                                archetype_cluster_index=cluster_index,
                                archetype_cluster_count=len(archetype_blocks),
                            ),
                        )
                    )

    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_id((assignments_path, summary_path, metadata_path))
    pdf = canvas.Canvas(str(output), pagesize=(PAGE_W, PAGE_H), invariant=1, pageCompression=1)
    pdf.setTitle("ModeNova all-category public archetype review")
    pdf.setAuthor("ModeNova")
    total_pages = len(page_specs)
    missing_count = 0
    source_counts: Counter[str] = Counter()
    rendered_brand_keys: list[tuple[str, int, str]] = []

    card_w = (PAGE_W - 2 * MARGIN_X - (GRID_COLS - 1) * GAP_X) / GRID_COLS
    card_h = (GRID_TOP - GRID_BOTTOM - (GRID_ROWS - 1) * GAP_Y) / GRID_ROWS
    for page_number, (kind, payload) in enumerate(page_specs, start=1):
        if kind == "cover":
            draw_cover(pdf, blocks, total_pages, snapshot)
        elif kind == "category":
            category = str(payload)
            draw_category_divider(pdf, category, [b for b in blocks if b.category == category])
            draw_footer(pdf, page_number, total_pages, snapshot)
        elif kind == "archetype":
            category, public_name, archetype_blocks = payload
            draw_archetype_divider(pdf, category, public_name, archetype_blocks)
            draw_footer(pdf, page_number, total_pages, snapshot)
        else:
            page = payload
            assert isinstance(page, ClusterPage)
            draw_cluster_header(pdf, page)
            for index, row in enumerate(page.rows):
                grid_row, grid_col = divmod(index, GRID_COLS)
                x = MARGIN_X + grid_col * (card_w + GAP_X)
                y = GRID_TOP - (grid_row + 1) * card_h - grid_row * GAP_Y
                image_ref = resolve_moodboard(row, metadata)
                source_counts[image_ref.source] += 1
                missing_count += int(
                    draw_brand_card(
                        pdf,
                        row,
                        image_ref,
                        cache_dir,
                        x,
                        y,
                        card_w,
                        card_h,
                    )
                )
                rendered_brand_keys.append(
                    (
                        page.block.category,
                        cluster_number(row.get("cluster")),
                        key_text(row.get("brand_name")),
                    )
                )
            draw_footer(pdf, page_number, total_pages, snapshot)
        pdf.showPage()
    pdf.save()

    expected_keys = [
        (block.category, cluster_number(row.get("cluster")), key_text(row.get("brand_name")))
        for block in blocks
        for row in block.rows
    ]
    if Counter(rendered_brand_keys) != Counter(expected_keys):
        raise RuntimeError("Coverage validation failed: PDF did not render every selected assignment once")
    return {
        "output": str(output),
        "snapshot": snapshot,
        "pages": total_pages,
        "categories": len(selected_categories),
        "clusters": len(blocks),
        "brands": len(rendered_brand_keys),
        "missing_images": missing_count,
        "fallback_images": sum(
            count for source, count in source_counts.items() if source not in {"metadata", "missing"}
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--category", action="append", default=[], help="Category to include; repeatable")
    parser.add_argument(
        "--cluster",
        action="append",
        default=[],
        type=parse_cluster_spec,
        metavar="CATEGORY:ID",
        help="Exact technical cluster to include; repeatable",
    )
    parser.add_argument("--max-clusters", type=int, help="Limit after deterministic sorting (for QA only)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_pdf(
        output=args.output.resolve(),
        assignments_path=args.assignments.resolve(),
        summary_path=args.summary.resolve(),
        metadata_path=args.metadata.resolve(),
        cache_dir=args.cache_dir.resolve(),
        categories=args.category,
        clusters=args.cluster,
        max_clusters=args.max_clusters,
    )
    print("Cluster review PDF built")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
