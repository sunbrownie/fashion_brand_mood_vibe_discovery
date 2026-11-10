"""Customer-facing archetype copy and deterministic mapping for taste profiles.

Technical category clusters remain unchanged.  Clothes clusters that Ellina's
review split are resolved per brand with reviewed overrides plus a documented
dominant default for the rest of the cluster.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


ARCHETYPES = (
    (1, "The Expedition", "Outdoor, technical, adventure", "You like knowing your clothes are built for anything. Technical, functional, ready for the elements — even if the furthest your jacket has made it so far is the local coffee shop. The wilderness can wait. You’ll be ready when it calls."),
    (2, "The Understated", "Quiet luxury, heritage minimalism", "You like the label on the inside. Quiet luxury at its best — no logo needed to make an impact. You’re building a timeless wardrobe slowly, intentionally, and it shows."),
    (3, "The Disruptor", "Avant-garde, conceptual, challenging", "You like clothes that start a conversation you didn’t ask for. A plain t-shirt and trousers? Never. For you, getting dressed is a statement — sometimes art, always intentional."),
    (4, "The Dreamer", "Romantic, whimsical, feminine softness", "You like floaty sleeves, soft prints, and the kind of dress that makes you feel like you’re in a film. When in doubt, a lace detail, a cute bag, a ribbon or a pearl will always save the day."),
    (5, "The Underground", "Street, skate, subculture", "Your clothes are how you express yourself. You liked the brands before they were cool, and even if the underground lifestyle is more of a memory now, the attitude never left your wardrobe."),
    (6, "Modern Minimalist", "Considered, edited, contemporary clean", "You want everything to go with everything. Clean silhouettes, simple colours, easy to dress for any occasion — office to dinner to a night out with barely a second thought."),
    (7, "After Midnight", "Dark, directional, sensual, conceptual", "Dark, conceptual, sensual — all could be you. You’re drawn to intriguing silhouettes, dark palettes, and the kind of outfit that makes people look twice without knowing why."),
    (8, "The Trove", "Maximalist, bold print, eclectic collector", "More is more in your wardrobe. Colours, patterns, textures — the whole lot. Your wardrobe looks like a very good vintage market, and that’s entirely intentional."),
    (9, "Sun Bleached", "Coastal slow, linen, washed ease", "You like linen that wrinkles, mornings that stretch, and clothes that feel like being on holiday even when you’re not."),
    (10, "The Workshop", "Heritage craft, workwear, selvedge", "You like things that are built to last. You’ve never thrown away a good leather jacket and you never will."),
    (11, "The Dresser", "Polished occasion, smart-dressed", "You like getting dressed up on purpose. Any occasion is the right occasion if you dress for it."),
    (12, "Endorphin", "Active, sport, movement, wellness", "You like looking good at 6am. Gym kit is a whole category in your wardrobe and you’re not sorry."),
    (13, "The Wanderer", "Global artisan, bohemian, travel craft", "You like clothes that come with a story. Handmade, artisanal, rooted in craft and culture from somewhere far away — your wardrobe is a collection of places, not just pieces."),
    (14, "The Alumni", "Classic casual, prep, wardrobe uniform", "You treat your outfits like a uniform. Work, uni, a quick padel game, coffee with friends — all perfect occasions for your signature look."),
    (15, "Riviera", "Resort, coastal luxury, Mediterranean", "You like being somewhere warm, or at least dressing like you are. Light materials, colour, prints, and a glass of something cold always nearby."),
)

ARCHETYPE_BY_NAME = {
    name: {
        "public_cluster_id": f"archetype-{number:02d}",
        "public_cluster_number": number,
        "public_cluster_name": name,
        "public_cluster_core_aesthetic": core,
        "public_cluster_description": description,
    }
    for number, name, core, description in ARCHETYPES
}

# Dominant customer-facing label for every current numeric category cluster.
# These are display mappings only; the numeric cluster assignments are stable.
CLUSTER_PUBLIC_NAMES = {
    "bags": ("The Workshop", "The Underground", "The Understated", "The Expedition", "The Disruptor", "Modern Minimalist", "The Expedition", "The Wanderer", "Sun Bleached", "After Midnight", "Riviera", "The Trove", "The Dresser", "Modern Minimalist", "The Dresser", "The Wanderer", "The Dresser"),
    "clothes": ("Sun Bleached", "The Dreamer", "The Dresser", "The Workshop", "Riviera", "The Dresser", "Modern Minimalist", "Modern Minimalist", "Modern Minimalist", "The Expedition", "The Wanderer", "The Alumni", "The Disruptor", "The Underground", "The Underground", "The Expedition", "Modern Minimalist"),
    "jewellery": ("The Understated", "The Understated", "The Underground", "The Dresser", "The Wanderer", "The Wanderer", "The Understated", "The Alumni", "The Expedition", "The Trove", "The Disruptor", "Modern Minimalist", "The Dresser", "Riviera", "After Midnight", "The Trove"),
    "shoes": ("Modern Minimalist", "The Dresser", "The Dresser", "The Dreamer", "The Underground", "The Underground", "The Alumni", "Endorphin", "The Understated", "Modern Minimalist", "The Disruptor", "The Expedition", "Sun Bleached", "The Expedition", "The Workshop", "The Dresser", "The Dresser", "Riviera", "The Dreamer", "The Alumni", "The Alumni", "After Midnight"),
    "swimwear_lingerie": ("The Wanderer", "After Midnight", "Riviera", "Modern Minimalist", "Riviera"),
}


def normalise(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = text.replace("&", " and ").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _names(*values: str) -> set[str]:
    return {normalise(value) for value in values}


# Brand-level decisions for the mixed live clothes clusters.  Brands not listed
# inherit the documented dominant label in CLUSTER_PUBLIC_NAMES.
CLOTHES_PUBLIC_OVERRIDES = {
    0: {
        "Endorphin": _names("ALO YOGA", "Beyond Yoga", "Deiji Studios", "Gymshark", "NAGNATA", "Sweaty Betty", "Varley", "Vuori"),
    },
    1: {
        "The Trove": _names("Anna Sui", "DVF", "Diane von Furstenberg", "Farm Rio", "Hofmann Copenhagen", "La Double J", "Lisou", "Mary Katrantzou", "MSGM", "Pucci", "Rich Mnisi", "Stine Goya", "Tongoro", "Traffic People", "Trina Turk"),
    },
    2: {
        "The Understated": _names("Bally", "Barneys Originals", "Belstaff", "Celine", "Cernucci", "Chanel", "Hermès", "Hermes", "Max Mara", "Nanushka", "Nour Hammour", "Saint Laurent", "Tod's", "Tods"),
    },
    5: {
        "The Trove": _names("Borgo De Nor", "Clio Peppiatt", "Cult Gaia", "DAVID KOMA", "DIMA AYAD", "Mirror Palais"),
        "The Dreamer": _names("Cecilie Bahnsen", "LoveShackFancy", "Simone Rocha"),
    },
    7: {
        "The Dresser": _names("Balmain", "Favourbrook", "Karl Lagerfeld", "Polo Ralph Lauren", "Thom Sweeney"),
    },
    8: {
        "The Dreamer": _names("DISSH", "Djerf Avenue", "ROTATE", "ROTATE Birger Christensen", "Jonathan Simkhai", "Significant Other", "Sessun", "TWIST & TANGO", "Twist & Tango"),
    },
    10: {
        "Sun Bleached": _names("Ciao Lucia", "Daydreamer LA", "The Handloom", "Handloom", "Los Angeles Allow"),
    },
    13: {
        "Endorphin": _names("ADANOLA", "Aim'n", "Bandit Running", "Castore", "Girlfriend Collective", "Gymshark", "Lululemon", "Nike", "Oner Active", "P.E Nation", "Sweaty Betty"),
        "The Expedition": _names("Aztech Mountain", "Burton", "Fusalp", "K WAY", "K-Way", "Perfect Moment"),
        "The Alumni": _names("Fred Perry", "Gant", "Lacoste"),
    },
    15: {
        "After Midnight": _names("A-COLD-WALL*", "A Cold Wall", "Han Kjøbenhavn", "Han Kjobenhavn", "Helmut Lang", "Mackintosh", "Nanamica", "OAMC", "SOIA & KYO", "Wardrobe NYC"),
        "Modern Minimalist": _names("Arket", "Aspesi", "Kloke", "Monique van Heist", "Norse Projects", "Studio Nicholson"),
        "The Workshop": _names("Alex Mill", "Arpenteur", "Ben Davis", "Dickies", "Kestin", "Knickerbocker NYC", "Private White V.C.", "Private White Vc", "Taylor Stitch", "Universal Works"),
    },
    16: {
        "Sun Bleached": _names("Aiayu", "Alex Crane", "Bite Studios", "Black Crane", "HHPLIFT", "Knowledge Cotton", "knowledge cotton apparel", "Mate the Label", "Ninety Percent", "Nobody's Child", "Outerknown", "Tentree", "Thought Clothing", "Tradlands", "World of Crow"),
        "The Wanderer": _names("Caralarga", "Doodlage", "Story Mfg"),
    },
}

GLOBAL_PUBLIC_OVERRIDES = {
    ("clothes", normalise("The Attico")): "After Midnight",
    ("bags", normalise("The Attico")): "After Midnight",
    ("bags", normalise("Alexander Wang")): "After Midnight",
    ("shoes", normalise("Comme des Garçons")): "The Disruptor",
    ("jewellery", normalise("Comme des Garçons")): "The Disruptor",
    ("bags", normalise("ALO YOGA")): "Endorphin",
    ("jewellery", normalise("ALO YOGA")): "Endorphin",
    ("shoes", normalise("ALO YOGA")): "Endorphin",
    ("clothes", normalise("A Bathing Ape")): "The Underground",
    ("clothes", normalise("Mizuno")): "Endorphin",
    ("bags", normalise("MOWALOLA")): "The Disruptor",
    ("jewellery", normalise("Issey Miyake")): "The Disruptor",
    ("jewellery", normalise("Henrik Vibskov")): "The Disruptor",
    ("jewellery", normalise("Marni")): "The Disruptor",
    ("shoes", normalise("Burton")): "The Expedition",
    ("shoes", normalise("Fusalp")): "The Expedition",
    ("shoes", normalise("K WAY")): "The Expedition",
    ("shoes", normalise("K-Way")): "The Expedition",
    ("shoes", normalise("Herschel")): "The Expedition",
    ("swimwear_lingerie", normalise("Baserange")): "Modern Minimalist",
    ("swimwear_lingerie", normalise("SKIMS")): "Modern Minimalist",
    ("swimwear_lingerie", normalise("Von Dutch")): "The Underground",
    ("clothes", normalise("Melie Bianco")): "The Understated",
}


def public_name_for_brand(category: str, cluster: int, brand_name: str) -> tuple[str, str]:
    try:
        default_name = CLUSTER_PUBLIC_NAMES[category][cluster]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"No public archetype mapping for {category}-{cluster}") from exc
    normalized = normalise(brand_name)
    reviewed_override = GLOBAL_PUBLIC_OVERRIDES.get((category, normalized))
    if reviewed_override:
        return reviewed_override, "reviewed_brand_override"
    if category != "clothes":
        return default_name, "cluster_mapping"
    for public_name, brands in CLOTHES_PUBLIC_OVERRIDES.get(cluster, {}).items():
        if normalized in brands:
            return public_name, "reviewed_brand_override"
    method = "dominant_cluster_default" if cluster in CLOTHES_PUBLIC_OVERRIDES else "direct_cluster_mapping"
    return default_name, method


def enrich_public_archetypes(data_dir: Path) -> None:
    assignments_path = data_dir / "ios_brand_cluster_assignments.csv"
    assignments = pd.read_csv(assignments_path)
    original_identity = assignments[["brand_name", "category", "cluster", "cluster_id"]].copy()

    public_rows = []
    for row in assignments.itertuples(index=False):
        public_name, method = public_name_for_brand(str(row.category), int(row.cluster), str(row.brand_name))
        public = ARCHETYPE_BY_NAME[public_name]
        public_rows.append({**public, "public_mapping_method": method})
    public_frame = pd.DataFrame(public_rows, index=assignments.index)
    for column in public_frame.columns:
        assignments[column] = public_frame[column]

    if not original_identity.equals(assignments[["brand_name", "category", "cluster", "cluster_id"]]):
        raise RuntimeError("Public archetype enrichment changed technical cluster identity")
    if assignments["public_cluster_name"].eq("").any() or assignments["public_cluster_description"].eq("").any():
        raise RuntimeError("Public archetype enrichment left blank customer-facing copy")

    assignments.to_csv(assignments_path, index=False)
    assignments.to_csv(data_dir / "ios_brand_cluster_assignments.tsv", index=False, sep="\t")

    public_assignments = assignments[[
        "brand_name", "category", "cluster", "cluster_id", "public_cluster_id",
        "public_cluster_number", "public_cluster_name", "public_cluster_description",
        "public_mapping_method",
    ]].copy()
    public_assignments.to_csv(data_dir / "brand_public_cluster_assignments.csv", index=False)

    join_columns = [
        "brand_name", "category", "public_cluster_id", "public_cluster_number",
        "public_cluster_name", "public_cluster_description", "public_mapping_method",
    ]
    public_join = assignments[join_columns].copy()
    umap_path = data_dir / "ios_brand_umap.csv"
    umap = pd.read_csv(umap_path)
    technical_umap = umap[["brand_name", "category", "cluster", "cluster_id", "x", "y", "umap_x", "umap_y"]].copy()
    umap = umap.drop(columns=[column for column in join_columns[2:] if column in umap.columns])
    umap = umap.merge(public_join, on=["brand_name", "category"], how="left", validate="one_to_one")
    if umap["public_cluster_id"].isna().any():
        raise RuntimeError("UMAP public archetype join has missing rows")
    if not technical_umap.equals(umap[technical_umap.columns]):
        raise RuntimeError("Public archetype enrichment changed UMAP coordinates or numeric clusters")
    umap.to_csv(umap_path, index=False, float_format="%.8f")
    umap.to_csv(data_dir / "ios_brand_umap.tsv", index=False, sep="\t", float_format="%.8f")

    summary_path = data_dir / "ios_cluster_summary.csv"
    summary = pd.read_csv(summary_path)
    dominant_ids, dominant_names, dominant_descriptions, statuses = [], [], [], []
    for row in summary.itertuples(index=False):
        name = CLUSTER_PUBLIC_NAMES[str(row.category)][int(row.cluster)]
        public = ARCHETYPE_BY_NAME[name]
        dominant_ids.append(public["public_cluster_id"])
        dominant_names.append(name)
        dominant_descriptions.append(public["public_cluster_description"])
        statuses.append("split_with_brand_overrides" if str(row.category) == "clothes" and int(row.cluster) in CLOTHES_PUBLIC_OVERRIDES else "cluster_mapping")
    summary["dominant_public_cluster_id"] = dominant_ids
    summary["dominant_public_cluster_name"] = dominant_names
    summary["dominant_public_cluster_description"] = dominant_descriptions
    summary["public_mapping_status"] = statuses
    summary.to_csv(summary_path, index=False)

    archetype_frame = pd.DataFrame([
        {
            "public_cluster_id": f"archetype-{number:02d}",
            "public_cluster_number": number,
            "public_cluster_name": name,
            "public_cluster_core_aesthetic": core,
            "public_cluster_description": description,
        }
        for number, name, core, description in ARCHETYPES
    ])
    archetype_frame.to_csv(data_dir / "public_cluster_archetypes.csv", index=False)

    summary_rows = []
    for (category, public_id), group in assignments.groupby(["category", "public_cluster_id"], sort=False):
        first = group.iloc[0]
        source_cluster_ids = sorted(
            group["cluster_id"].astype(str).unique(),
            key=lambda value: (value.rsplit("-", 1)[0], int(value.rsplit("-", 1)[1])),
        )
        summary_rows.append({
            "category": category,
            "public_cluster_id": public_id,
            "public_cluster_number": int(first["public_cluster_number"]),
            "public_cluster_name": first["public_cluster_name"],
            "public_cluster_description": first["public_cluster_description"],
            "n_brands": len(group),
            "source_cluster_ids": ", ".join(source_cluster_ids),
            "example_brands": ", ".join(group["brand_name"].astype(str).head(8)),
        })
    pd.DataFrame(summary_rows).sort_values(
        ["category", "public_cluster_number"]
    ).to_csv(data_dir / "ios_public_cluster_summary.csv", index=False)

    anchors = {
        ("clothes", "alo yoga"): "Endorphin",
        ("clothes", "marine layer"): "Sun Bleached",
        ("clothes", "pucci"): "The Trove",
        ("clothes", "carhartt wip"): "The Workshop",
        ("clothes", "a cold wall"): "After Midnight",
    }
    lookup = {
        (str(row.category), normalise(str(row.brand_name))): str(row.public_cluster_name)
        for row in assignments.itertuples(index=False)
    }
    for key, expected in anchors.items():
        if key in lookup and lookup[key] != expected:
            raise RuntimeError(f"Public archetype anchor mismatch for {key}: {lookup[key]} != {expected}")
