#!/usr/bin/env python3
"""Fully integrate approved V2 catalog additions into data, embeddings and clusters."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from build_generation_queue import SPECIAL_PROMPT_OVERRIDES


ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "recommender" / "hosting_bundle"
DATA = HOST / "data"
APP_ASSETS = ROOT / "ios" / "BrandMoodboardFinder" / "BrandMoodboardFinder" / "BrandMoodboards"
OVERLAY = APP_ASSETS / "v2_catalog_additions.json"
V2_MANIFEST = APP_ASSETS / "v2_manifest.json"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CATEGORY_ORDER = {name: index for index, name in enumerate(("clothes", "shoes", "bags", "jewellery", "swimwear_lingerie"))}

EXPANSION_PROMPTS = {
    ("shoes", "New Balance"): (
        "New Balance footwear balances grey heritage running, technical silver runners, sculptural lifestyle sneakers, "
        "retro court shoes, colourful racing shoes and rugged trail shoes."
    ),
    ("shoes", "Nike"): (
        "Nike footwear spans visible-air lifestyle shoes, cushioned road running, retro court icons, basketball, "
        "rugged ACG trail shoes and experimental sport design."
    ),
    ("shoes", "Puma"): (
        "Puma footwear combines motorsport driving shoes, terrace classics, suede icons, sculptural fashion sneakers, "
        "performance running and football boots."
    ),
    ("shoes", "Burton"): (
        "Burton Menswear London footwear combines polished leather loafers, brogues and Derby shoes with refined "
        "Chelsea and desert boots plus clean smart-casual trainers."
    ),
}

CATEGORY_MATERIALS = {
    "clothes": "cotton, wool, denim, leather, knit and technical seasonal fabrics",
    "shoes": "leather, suede, canvas, mesh, rubber and technical textiles",
    "bags": "leather, suede, canvas, nylon, woven fibres and metal hardware",
    "jewellery": "gold, silver, enamel, pearls, crystals and gemstones",
    "swimwear_lingerie": "stretch jersey, technical swim fabric, lace and soft elastic",
}

CLUSTER_OVERRIDES = {
    ("shoes", "new balance"): 7,
    ("shoes", "nike"): 7,
    ("shoes", "puma"): 7,
}

# Centroid similarity is useful for surfacing candidates, but some mixed-product
# boards need a final visual judgement.  These assignments were reviewed against
# the accepted V2 image and the stable cluster vocabulary after the descriptions
# below were rewritten.
CLUSTER_REVIEW_OVERRIDES = {
    ("bags", "casablanca"): 4,
    ("bags", "chopova lowena"): 1,
    ("bags", "cult gaia"): 4,
    ("clothes", "girlfriend collective"): 13,
    ("clothes", "house of sunny"): 1,
    ("clothes", "karoline vitto"): 2,
    ("clothes", "rotate birger christensen"): 8,
    ("clothes", "weekday"): 13,
    ("shoes", "coperni"): 2,
    ("shoes", "jacquemus"): 2,
    ("shoes", "burton"): 16,
}

NEW_CLUSTER_ASSIGNMENT_KEYS = {
    ("shoes", "comme des garcons play"),
}

INVALID_TASTE_ROWS = {
    ("clothes", name)
    for name in {
        "acacia", "adriana degreas", "agua bendita", "andrea iyamah", "araks", "bather", "bluebella",
        "bondi born", "carine gilson", "cdlp", "cou cou intimates", "fleur du mal", "for love and lemons",
        "frankies bikinis", "fruity booty", "hunza g", "jade swim", "kiki de montparnasse", "le petit trou",
        "left on friday", "maison close", "marysia", "matteau", "maygel coronel", "melissa odabash", "mikoh",
        "montce", "natori", "only hearts", "orlebar brown", "peony swimwear", "saxx", "skims",
        "solid and striped", "suboo", "wild lovers",
    }
}

# Human-reviewed metadata corrections for accepted V2 boards whose original text
# card no longer described the regenerated image accurately enough.
DESCRIPTION_OVERRIDES = {
    ("shoes", "Comme des Garçons PLAY"): {
        "aesthetic_keywords": "playful, graphic, logo-led, casual, pop-art",
        "silhouettes": "low-top canvas sneakers, high-top canvas sneakers, retro rubber-toe trainers",
        "materials": "cotton canvas, rubber foxing, embroidery and printed heart appliqué",
        "palette": "black, white, cream, navy and signature red",
    },
    ("bags", "Chopova Lowena"): {
        "aesthetic_keywords": "punk, folkloric, grunge, handcrafted, hardware-heavy",
        "silhouettes": "tartan totes, grommet bucket bags, asymmetric shoulder bags, chain-handle mini bags",
        "materials": "tartan wool, black leather, mesh, silver chains, studs and charms",
        "palette": "red, cobalt, forest green, black and silver",
    },
    ("clothes", "Made By Minga"): {
        "aesthetic_keywords": "artisanal, hand-knit, soft, natural, relaxed",
        "silhouettes": "chunky sweaters, balloon-sleeve cardigans, crochet hats, scarves and embroidered lounge sets",
        "materials": "alpaca and wool knits, crochet, soft cotton and hand embroidery",
        "palette": "oatmeal, cream, camel, chocolate brown and dusty blue",
    },
    ("jewellery", "Simon Miller"): {
        "aesthetic_keywords": "playful, resort, novelty, colourful, handcrafted",
        "silhouettes": "oversized palm-tree drops, fruit earrings, starfish earrings and raffia statement drops",
        "materials": "resin, enamel, raffia, beads and gold-tone findings",
        "palette": "emerald green, orange, yellow, natural raffia, black and gold",
    },
    ("jewellery", "Yellow Swallow"): {
        "aesthetic_keywords": "edgy, gothic, sad-girl, subcultural, Korean indie",
        "silhouettes": "leather-and-pearl chokers, oxidised chain pendants, crystal crosses, spider brooches, sharp hair clips and blackened floral jewellery",
        "materials": "oxidised silver, gunmetal, black nickel, smoky crystals, pearls, black leather and distressed cotton",
        "palette": "black, gunmetal, dirty white, smoky crystal, muted mauve and burgundy",
    },
    ("bags", "CASABLANCA"): {
        "aesthetic_keywords": "sport-luxe, sunny, witty, retro, graphic",
        "silhouettes": "tennis-racket crossbodies, orange-shaped top handles, travel pouches, circular sport bags and bowling bags",
        "materials": "smooth leather, coated canvas, enamel details and polished hardware",
        "palette": "tennis green, orange, sky blue, white and black",
    },
    ("clothes", "Conner Ives"): {
        "aesthetic_keywords": "upcycled glamour, eclectic Americana, romantic, theatrical, crafted",
        "silhouettes": "reconstructed slip dresses, printed chiffon gowns, tuxedo shirting, sequinned evening dresses and embellished skirts",
        "materials": "reclaimed jersey, chiffon, lace, sequins, velvet and leather",
        "palette": "cream, black, oxblood, antique gold and vintage mixed prints",
    },
    ("clothes", "Weekday"): {
        "aesthetic_keywords": "urban, utilitarian, oversized, grungy, youthful",
        "silhouettes": "cropped bombers, oversized knitwear, printed mesh tops, cargo skirts, wide trousers and relaxed denim",
        "materials": "washed denim, technical nylon, wool knit, mesh and cotton twill",
        "palette": "charcoal, black, olive, stone, washed grey and muted purple",
    },
    ("clothes", "Bimba Y Lola"): {
        "aesthetic_keywords": "polished, eclectic, feminine, modern, softly sculptural",
        "silhouettes": "leather tailoring, crisp shirts, draped midi dresses, lace skirts, wide trousers and soft knitwear",
        "materials": "leather, silk-like satin, lace, wool knit and tailored twill",
        "palette": "chocolate, cream, olive, burgundy, navy, powder blue and blush",
    },
    ("shoes", "Coperni"): {
        "aesthetic_keywords": "futuristic, sleek, sculptural, sensual, high-shine",
        "silhouettes": "cut-out slingbacks, wedge ankle boots, metallic pumps, transparent wedges, spiral sandals and aerodynamic flats",
        "materials": "smooth leather, patent leather, mirrored metallic leather, transparent vinyl and technical mesh",
        "palette": "black, white, silver, transparent crystal and signal red",
    },
    ("clothes", "Desigual"): {
        "aesthetic_keywords": "modern eclectic, artful, colourful, urban, patchworked",
        "silhouettes": "asymmetric knit dresses, liquid tops, embroidered jackets, patchwork wide jeans, floral cardigans and printed tailoring",
        "materials": "knit, coated jersey, embroidered wool, denim, faux fur and printed satin",
        "palette": "black, red, cobalt, burgundy, denim blue and multicolour florals",
    },
    ("clothes", "Shangri-la Heritage"): {
        "aesthetic_keywords": "motorcycle, heritage, rugged, handcrafted, mountain-inspired",
        "silhouettes": "leather motorcycle jackets, shearling flight jackets, western layers and sturdy workwear",
        "materials": "aged leather, shearling, waxed cotton, denim and metal hardware",
        "palette": "tobacco, chocolate brown, black, cream and faded indigo",
    },
    ("shoes", "Geox"): {
        "aesthetic_keywords": "practical, breathable, polished, versatile, comfort-led",
        "silhouettes": "technical sneakers, waterproof ankle boots, pumps, slip-ons, runners and sport sandals",
        "materials": "perforated leather, suede, breathable mesh, waterproof technical fabric and rubber soles",
        "palette": "silver, navy, burgundy, cream, teal and coral",
    },
    ("clothes", "No Problemo"): {
        "aesthetic_keywords": "graphic streetwear, retro sci-fi, skate, playful, utilitarian",
        "silhouettes": "logo short-sleeve T-shirts, logo crewneck sweatshirts, striped long sleeves, ripstop workwear, technical shells, fleece jackets and silver puffers",
        "materials": "cotton jersey, brushed fleece, ripstop cotton, sherpa fleece and quilted technical fabric",
        "palette": "black, washed grey, forest green, cream, silver and fluorescent yellow accents",
    },
    ("bags", "Marine Serre"): {
        "aesthetic_keywords": "futuristic, crescent-moon, polished, graphic, upcycled-luxe",
        "silhouettes": "crescent hobos, moon-print totes, quilted chain bags, metallic shoulder bags and cylindrical top handles",
        "materials": "smooth leather, moon-print coated canvas, quilted leather, metallic leather and silver hardware",
        "palette": "black, red, tan, silver and cobalt blue",
    },
    ("bags", "Cult Gaia"): {
        "aesthetic_keywords": "sculptural, resort, surreal, statement, architectural",
        "silhouettes": "bamboo ark bags, marbled acrylic clutches, pearl sphere bags, shell clutches, curved hobos and crystal pouches",
        "materials": "bamboo, marbled acrylic, pearlescent resin, gold-tone metal and crystals",
        "palette": "natural bamboo, emerald, pearl, gold, ivory and crystal",
    },
    ("bags", "ALO YOGA"): {
        "aesthetic_keywords": "quiet luxury, wellness-luxe, refined, minimal, polished",
        "silhouettes": "leather duffles, drawstring shoulder bags, perforated totes, bowling bags and bucket bags",
        "materials": "smooth leather, perforated leather, suede, gold hardware and crystal-like charms",
        "palette": "espresso, cream, black, chocolate brown and warm gold",
    },
    ("bags", "Miu Miu"): {
        "aesthetic_keywords": "playful luxury, colourful, youthful, polished, tactile",
        "silhouettes": "perforated hobos, compact top handles, matelassé shoulder bags, chain bags and gathered clutches",
        "materials": "perforated leather, smooth leather, matelassé leather and gold hardware",
        "palette": "red, cobalt, pink, forest green, yellow and black",
    },
    ("bags", "Maje"): {
        "aesthetic_keywords": "Parisian, feminine, bohemian, polished, tactile",
        "silhouettes": "fringed M shoulder bags, compact crossbodies, soft hobos and structured mini bags",
        "materials": "suede, smooth leather, woven leather, fringe and gold hardware",
        "palette": "black, cognac, cream, burgundy and muted jewel tones",
    },
    ("bags", "Marge Sherwood"): {
        "aesthetic_keywords": "retro-modern, playful, sleek, youthful, sculptural",
        "silhouettes": "curved shoulder bags, east-west bags, soft hobos, compact top handles and charm-decorated minis",
        "materials": "glossy leather, suede, grained leather and silver-tone charms",
        "palette": "black, chocolate, burgundy, cream and bright seasonal colour",
    },
    ("clothes", "Girlfriend Collective"): {
        "aesthetic_keywords": "inclusive activewear, sustainable, clean, warm, everyday",
        "silhouettes": "leggings, bike shorts, sports bras, unitards, tennis dresses, track sets and relaxed sweats",
        "materials": "recycled performance knit, stretch jersey, fleece and ribbed technical fabric",
        "palette": "earthy sage, slate blue, plum, mustard, terracotta, cream and black",
    },
    ("clothes", "House Of Sunny"): {
        "aesthetic_keywords": "playful, retro-futurist, colourful, youthful, knit-led",
        "silhouettes": "graphic cardigans, swirled knit dresses, cropped knits, halter dresses and flared co-ordinates",
        "materials": "recycled knit, ribbed jersey, crochet and faux leather",
        "palette": "sage, cobalt, lilac, chocolate, cream and vivid multicolour patterns",
    },
    ("clothes", "Karoline Vitto"): {
        "aesthetic_keywords": "body-positive, sculptural, sensual, precise, metal-accented",
        "silhouettes": "cut-out dresses, one wire-frame bra look, curved metal-frame tops and fitted skirts",
        "materials": "stretch jersey, polished metal wire, lycra and soft tailoring fabric",
        "palette": "black, espresso, oxblood, silver and one saturated colour accent",
    },
    ("clothes", "Obey"): {
        "aesthetic_keywords": "streetwear, skate, graphic, political, mixed-gender",
        "silhouettes": "logo T-shirts, graphic sweatshirts, work jackets, loose trousers, caps and casual layers",
        "materials": "cotton jersey, fleece, canvas, twill and washed denim",
        "palette": "black, ecru, olive, red, navy and graphic colour",
    },
    ("clothes", "Opera Sport"): {
        "aesthetic_keywords": "Copenhagen cool, sporty, feminine, minimal, contemporary",
        "silhouettes": "track jackets, drawstring trousers, sleek dresses, striped knits and relaxed tailored layers",
        "materials": "recycled technical fabric, cotton, wool knit and soft jersey",
        "palette": "black, cream, navy, burgundy and clear primary accents",
    },
    ("shoes", "New Balance"): {
        "aesthetic_keywords": "technical running, heritage sport, understated, performance, lifestyle",
        "silhouettes": "grey heritage runners, silver mesh sneakers, sculptural lifestyle trainers, retro court shoes, racing shoes and trail shoes",
        "materials": "suede, technical mesh, synthetic overlays, rubber and reflective details",
        "palette": "grey, silver, cream, navy, cobalt and bright racing accents",
    },
    ("shoes", "Nike"): {
        "aesthetic_keywords": "performance, iconic, experimental, street-sport, technical",
        "silhouettes": "visible-air sneakers, road runners, retro court shoes, basketball shoes, ACG trail shoes and experimental trainers",
        "materials": "engineered mesh, leather, suede, foam, rubber and technical synthetics",
        "palette": "black, white, silver, volt, orange and bold team colour",
    },
    ("shoes", "Puma"): {
        "aesthetic_keywords": "motorsport, terrace, retro sport, performance, fashion-forward",
        "silhouettes": "driving shoes, suede terrace sneakers, retro court shoes, sculptural fashion trainers, runners and football boots",
        "materials": "suede, leather, technical mesh, synthetic overlays and rubber",
        "palette": "red, forest green, black, silver, neon yellow and white",
    },
    ("shoes", "Burton"): {
        "aesthetic_keywords": "polished, classic, smart-casual, accessible, British menswear",
        "silhouettes": "penny and tassel loafers, brogues, Derby and Oxford shoes, Chelsea and desert boots, clean low-top trainers",
        "materials": "smooth and textured leather, suede, rubber soles and discreet metal hardware",
        "palette": "black, espresso brown, tan, cognac, oxblood, cream and subtle navy",
    },
    ("bags", "Anya Hindmarch"): {
        "aesthetic_keywords": "witty, playful, refined, inventive, characterful",
        "silhouettes": "eyes totes, cereal-box clutches, woven shoppers, structured top handles and embellished evening bags",
        "materials": "smooth leather, recycled nylon, woven fibres, sequins and appliqué",
        "palette": "black, red, cobalt, natural straw, metallics and bright novelty colour",
    },
    ("shoes", "Arizona Love"): {
        "aesthetic_keywords": "eclectic, bohemian, handcrafted, colourful, festival",
        "silhouettes": "bandana sandals, beaded platform sandals, fringed boots, embellished clogs and wrapped slides",
        "materials": "bandana cotton, suede, leather, beads, embroidery and fringe",
        "palette": "turquoise, red, tan, black, cream and multicolour textile print",
    },
    ("jewellery", "Valentino"): {
        "aesthetic_keywords": "romantic, polished, logo-led, modern, feminine",
        "silhouettes": "VLogo earrings, slim cuffs, chain bracelets, heart charms and restrained pendant necklaces",
        "materials": "gold-tone metal, enamel, crystals and leather details",
        "palette": "gold, black, red, ivory and pale pink",
    },
    ("clothes", "Carne Bollente"): {
        "aesthetic_keywords": "cheeky, sex-positive, graphic, playful, street-casual",
        "silhouettes": "embroidered T-shirts, graphic sweatshirts, easy shirts, casual knits and relaxed separates",
        "materials": "organic cotton jersey, fleece, knit and embroidery",
        "palette": "ecru, black, red, teal, sky blue and small bright accents",
    },
    ("clothes", "Lisa Says Gah"): {
        "aesthetic_keywords": "retro-romantic, playful, cottage-inspired, trend-led, feminine",
        "silhouettes": "printed midi dresses, romantic blouses, cardigans, flowing skirts and wide trousers",
        "materials": "cotton poplin, soft knit, satin, lace and printed viscose",
        "palette": "earthy neutrals, soft pastels, red, forest green and vintage multicolour print",
    },
    ("shoes", "Chopova Lowena"): {
        "aesthetic_keywords": "punk, rugged, folkloric, eclectic, hardware-heavy",
        "silhouettes": "tartan platform boots, studded motorcycle boots, trail shoes, clogs and hybrid lace-ups",
        "materials": "tartan wool, leather, suede, rubber, studs and carabiner hardware",
        "palette": "brown, black, forest green, red tartan and silver",
    },
    ("bags", "Coperni"): {
        "aesthetic_keywords": "futuristic, architectural, sleek, iconic, minimal",
        "silhouettes": "Swipe bags, glass-effect bags, compact top handles, curved shoulder bags and sculptural totes",
        "materials": "smooth leather, metallic leather, transparent acrylic and polished metal hardware",
        "palette": "black, white, silver, transparent and saturated accent colour",
    },
    ("shoes", "Jacquemus"): {
        "aesthetic_keywords": "sculptural, sensual, playful, Mediterranean, fashion-forward",
        "silhouettes": "asymmetric sandals, sculptural mules, curved heels, platform shoes and statement boots",
        "materials": "smooth leather, suede, raffia, transparent vinyl and polished metal",
        "palette": "cream, black, tan, yellow, red and bright seasonal colour",
    },
    ("bags", "JW Pei"): {
        "aesthetic_keywords": "modern, sculptural, colourful, accessible, playful",
        "silhouettes": "curved shoulder bags, top-handle minis, geometric totes, woven pouches and bowling bags",
        "materials": "vegan leather, recycled plastic, woven synthetic material and silver hardware",
        "palette": "orange, cobalt, lime, silver, black and cream",
    },
    ("clothes", "Sandy Liang"): {
        "aesthetic_keywords": "high-fashion girly, downtown, playful, nostalgic, polished",
        "silhouettes": "bow dresses, pleated skirts, fitted cardigans, cropped jackets and feminine tailoring",
        "materials": "satin, wool, cotton, lace, tweed and ribbon details",
        "palette": "black, navy, cream, ballet pink, red and pale blue",
    },
    ("clothes", "Patou"): {
        "aesthetic_keywords": "Parisian, playful, polished, feminine, modern",
        "silhouettes": "crisp shirts, voluminous skirts, tailored separates, bow details and clean day dresses",
        "materials": "cotton poplin, wool, technical taffeta, knit and smooth leather",
        "palette": "navy, white, black, red and soft pastel accents",
    },
    ("clothes", "The Attico"): {
        "aesthetic_keywords": "sensual, nightlife, sharp, glamorous, street-luxe",
        "silhouettes": "body-conscious dresses, oversized jackets, cut-out tops, cargo trousers and statement tailoring",
        "materials": "satin, leather, denim, sequins and technical nylon",
        "palette": "black, silver, red, acid brights and jewel tones",
    },
    ("shoes", "ATP Atelier"): {
        "aesthetic_keywords": "minimal, refined, Italian, practical, year-round",
        "silhouettes": "ankle boots, knee boots, loafers, sculptural sandals, mules and clean flats",
        "materials": "vegetable-tanned leather, suede, rubber and polished metal hardware",
        "palette": "black, cognac, cream, chocolate and muted seasonal colour",
    },
    ("clothes", "Kowtow"): {
        "aesthetic_keywords": "ethical, organic, architectural, relaxed, modern",
        "silhouettes": "sculptural dresses, boxy shirts, wide trousers, layered knitwear and utilitarian outerwear",
        "materials": "organic cotton, denim, wool knit and low-impact natural fibres",
        "palette": "navy, cobalt, black, cream and grounded seasonal colour",
    },
    ("clothes", "McQueen"): {
        "aesthetic_keywords": "sculptural, romantic, sharp, dramatic, modern",
        "silhouettes": "precise tailoring, draped dresses, corseted shapes, statement knitwear and engineered outerwear",
        "materials": "wool, leather, silk, lace, denim and polished metal details",
        "palette": "black, ivory, oxblood, cobalt, silver and vivid seasonal colour",
    },
    ("clothes", "ROTATE Birger Christensen"): {
        "aesthetic_keywords": "Copenhagen cool, sexy, sharp, playful, fashion-forward",
        "silhouettes": "oversized jackets with short shorts, rounded-sleeve tops, sleek dresses, wide trousers and modern separates",
        "materials": "tailored wool, satin, leather, jersey and restrained sequins",
        "palette": "black, chocolate, burgundy, silver and selective bright accents",
    },
}


def normalised(value: str) -> str:
    folded = "".join(
        character
        for character in unicodedata.normalize("NFKD", str(value).replace("&", " and "))
        if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[a-z0-9]+", folded.lower()))


def slugify(value: str) -> str:
    return normalised(value).replace(" ", "_")


def sentence_matching(prompt: str, prefix: str) -> str:
    for sentence in re.split(r"(?<=[.!?])\s+", prompt.strip()):
        if sentence.lower().startswith(prefix.lower()):
            return sentence.rstrip(".")
    return ""


def descriptors(entry: dict) -> dict[str, str]:
    key = (entry["category"], entry["brand_name"])
    prompt = EXPANSION_PROMPTS.get(key) or SPECIAL_PROMPT_OVERRIDES.get(key)
    if not prompt:
        raise RuntimeError(f"Missing integration description for {key}")
    first = re.split(r"(?<=[.!?])\s+", prompt.strip())[0]
    first = re.sub(r"^Create a vertical six-panel\s+", "", first, flags=re.I)
    first = re.sub(r"\s+moodboard\s*", " ", first, flags=re.I).strip(" .")
    silhouettes = sentence_matching(prompt, "Show ")
    silhouettes = re.sub(r"^Show\s+(six|eight)\s+distinct\s+", "", silhouettes, flags=re.I)
    palette = sentence_matching(prompt, "Use ")
    palette = re.sub(r"^Use\s+", "", palette, flags=re.I)
    return {
        "aesthetic_keywords": first,
        "silhouettes": silhouettes or first,
        "materials": CATEGORY_MATERIALS[entry["category"]],
        "palette": palette or "brand-signature neutrals and seasonal colour accents",
    }


def normalised_description_overrides() -> dict[tuple[str, str], dict[str, str]]:
    return {
        (category, normalised(brand_name)): values
        for (category, brand_name), values in DESCRIPTION_OVERRIDES.items()
    }


def sync_v2_moodboards_to_host() -> dict[tuple[str, str], str]:
    """Copy every accepted V2 asset, not only new catalog additions, into the HF bundle."""
    synced: dict[tuple[str, str], str] = {}
    entries = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))["entries"]
    for entry in entries:
        source = APP_ASSETS / Path(entry["ios_v2_path"]).name
        if not source.is_file():
            raise FileNotFoundError(source)
        slug = entry["slug"]
        category = entry["category"]
        target = DATA / "moodboards" / category / f"{slug}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        synced[(category, normalised(entry["brand_name"]))] = f"moodboards/{category}/{slug}.jpg"
    return synced


def build_catalog() -> tuple[pd.DataFrame, pd.DataFrame]:
    host_path = DATA / "brand_metadata.csv"
    host = pd.read_csv(host_path)
    synced_v2_paths = sync_v2_moodboards_to_host()
    overlay = json.loads(OVERLAY.read_text(encoding="utf-8"))["entries"]
    existing = {(row.category, normalised(row.brand_name)) for row in host.itertuples()}
    additions = []
    for entry in overlay:
        key = (entry["category"], normalised(entry["brand_name"]))
        if key in existing:
            continue
        fields = descriptors(entry)
        slug = slugify(entry["brand_name"])
        source = APP_ASSETS / entry["moodboard_asset_name"]
        target = DATA / "moodboards" / entry["category"] / f"{slug}.jpg"
        if not source.exists():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        additions.append({
            "brand_name": entry["brand_name"],
            "category": entry["category"],
            "official_website": entry["official_website"],
            "male": float(entry["male"]),
            "female": float(entry["female"]),
            "children": 0.0,
            "price": entry["price"],
            "moodboard_path": f"moodboards/{entry['category']}/{slug}.jpg",
            "slug": slug,
            "popularity": int(entry["popularity"]),
            "male_moodboard_path": "",
            **fields,
        })
        existing.add(key)
    host = pd.concat([host, pd.DataFrame(additions)], ignore_index=True, sort=False)
    override_lookup = normalised_description_overrides()
    overlay_lookup = {
        (entry["category"], normalised(entry["brand_name"])): entry
        for entry in overlay
    }

    play_key = ("shoes", normalised("Comme des Garçons PLAY"))
    host_keys = {(row.category, normalised(row.brand_name)) for row in host.itertuples()}
    if play_key not in host_keys:
        source = host[
            host["category"].eq("shoes")
            & host["brand_name"].map(normalised).eq(normalised("Comme des Garçons"))
        ].iloc[0]
        play_row = source.to_dict()
        play_row.update({
            "brand_name": "Comme des Garçons PLAY",
            "moodboard_path": synced_v2_paths[play_key],
            "slug": "comme_des_garcons_play",
            **override_lookup[play_key],
        })
        host = pd.concat([host, pd.DataFrame([play_row])], ignore_index=True, sort=False)

    for index, row in host.iterrows():
        key = (str(row["category"]), normalised(row["brand_name"]))
        if key in synced_v2_paths:
            host.at[index, "moodboard_path"] = synced_v2_paths[key]
        if key in overlay_lookup:
            entry = overlay_lookup[key]
            host.at[index, "official_website"] = entry["official_website"]
            host.at[index, "male"] = float(entry["male"])
            host.at[index, "female"] = float(entry["female"])
            host.at[index, "price"] = entry["price"]
            host.at[index, "popularity"] = int(entry["popularity"])
        if key in override_lookup:
            for column, value in override_lookup[key].items():
                host.at[index, column] = value
    host["_category_order"] = host["category"].map(CATEGORY_ORDER)
    host["_brand_order"] = host["brand_name"].astype(str).str.casefold()
    host = host.sort_values(["_category_order", "_brand_order"]).drop(columns=["_category_order", "_brand_order"])
    host = host.reset_index(drop=True)
    host.to_csv(host_path, index=False)

    prompt = (
        "Aesthetic: " + host["aesthetic_keywords"].fillna("").astype(str) + ". "
        "Silhouettes: " + host["silhouettes"].fillna("").astype(str) + ". "
        "Materials: " + host["materials"].fillna("").astype(str) + ". "
        "Palette: " + host["palette"].fillna("").astype(str) + "."
    )
    embedding_metadata = host[[
        "brand_name", "category", "official_website", "aesthetic_keywords", "silhouettes", "materials", "palette"
    ]].copy()
    embedding_metadata["prompt"] = prompt
    embedding_metadata["male"] = host["male"]
    embedding_metadata["female"] = host["female"]
    embedding_metadata["children"] = host["children"]
    embedding_metadata["price"] = host["price"]
    for path in (
        ROOT / "step_2_text_embeddings" / "brand_metadata.csv",
        ROOT / "step_2_text_embeddings" / "brand_metadata_today.csv",
    ):
        embedding_metadata.to_csv(path, index=False)
    return host, embedding_metadata


def update_source_csvs(host: pd.DataFrame) -> None:
    file_map = {
        "clothes": ROOT / "final_dataset" / "brands_clothes.csv",
        "shoes": ROOT / "final_dataset" / "brands_shoes.csv",
        "bags": ROOT / "final_dataset" / "brands_bags.csv",
        "jewellery": ROOT / "final_dataset" / "brands_jewellery.csv",
    }
    overlay = json.loads(OVERLAY.read_text(encoding="utf-8"))["entries"]
    overlay_keys = {(entry["category"], normalised(entry["brand_name"])) for entry in overlay}
    override_keys = set(normalised_description_overrides())
    managed_keys = overlay_keys | override_keys
    host_lookup = {(row.category, normalised(row.brand_name)): row for row in host.itertuples()}
    for category, path in file_map.items():
        frame = pd.read_csv(path)
        existing = {normalised(value) for value in frame["brand_name"]}
        for index, source_row in frame.iterrows():
            brand_key = normalised(source_row["brand_name"])
            key = (category, brand_key)
            if key not in managed_keys or key not in host_lookup:
                continue
            host_row = host_lookup[key]
            for column in ("official_website", "aesthetic_keywords", "silhouettes", "materials", "palette"):
                if column in frame.columns:
                    frame.at[index, column] = getattr(host_row, column)
        rows = []
        for overlay_category, brand_key in sorted(managed_keys):
            if overlay_category != category or brand_key in existing:
                continue
            row = host_lookup[(category, brand_key)]
            values = {
                "brand_name": row.brand_name,
                "official_website": row.official_website,
                "aesthetic_keywords": row.aesthetic_keywords,
                "silhouettes": row.silhouettes,
                "materials": row.materials,
                "palette": row.palette,
                "moodboard": f"{row.slug}.jpg",
                "male": int(row.male),
                "female": int(row.female),
                "price": row.price,
                "swimwear": 0,
                "lingerie": 0,
                "children": 0,
                "popularity": int(row.popularity),
            }
            rows.append({column: values.get(column, "") for column in frame.columns})
            existing.add(brand_key)
        if rows:
            frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
        frame = frame.sort_values("brand_name", key=lambda series: series.astype(str).str.casefold()).reset_index(drop=True)
        frame.to_csv(path, index=False)


def encode_text(metadata: pd.DataFrame) -> np.ndarray:
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        metadata["prompt"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    target = ROOT / "step_2_text_embeddings" / "brand_embeddings.npz"
    np.savez_compressed(target, embeddings=embeddings)
    shutil.copy2(target, DATA / "brand_embeddings.npz")
    return embeddings


def key_frame(frame: pd.DataFrame) -> dict[tuple[str, str], int]:
    return {
        (str(row.category), normalised(row.brand_name)): index
        for index, row in frame.iterrows()
    }


def update_projection_and_clusters(host: pd.DataFrame, embeddings: np.ndarray) -> None:
    taste_mask = pd.Series(
        [
            (row.category, normalised(row.brand_name)) not in INVALID_TASTE_ROWS
            for row in host.itertuples()
        ],
        index=host.index,
    )
    host = host.loc[taste_mask].reset_index(drop=True)
    embeddings = embeddings[taste_mask.to_numpy()]

    old_assignments = pd.read_csv(
        ROOT / "ios" / "BrandMoodboardFinder" / "BrandMoodboardFinder"
        / "ClusterData" / "ios_brand_cluster_assignments.csv"
    )
    summary = pd.read_csv(DATA / "ios_cluster_summary.csv")
    metadata_lookup = key_frame(host)
    old_lookup = {
        (str(row.category), normalised(row.brand_name)): int(row.cluster)
        for row in old_assignments.itertuples()
    }
    # Preserve the original pre-reassessment assignments across reproducible
    # reruns.  The first completed run writes this audit before the iOS exports
    # are refreshed, so it is the stable baseline for later manual review.
    reassessment_path = (
        ROOT / "step_3_moodboards" / "review" / "v2_cluster_reassessment.csv"
    )
    if reassessment_path.exists():
        baseline = pd.read_csv(reassessment_path)
        for baseline_row in baseline.itertuples():
            baseline_key = (
                str(baseline_row.category),
                normalised(str(baseline_row.brand_name)),
            )
            if pd.isna(baseline_row.old_cluster):
                old_lookup.pop(baseline_key, None)
                continue
            old_lookup[baseline_key] = int(baseline_row.old_cluster)
    for new_key in NEW_CLUSTER_ASSIGNMENT_KEYS:
        old_lookup.pop(new_key, None)
    summary_lookup = {
        (str(row.category), int(row.cluster)): row
        for row in summary.itertuples()
    }

    centroids: dict[tuple[str, int], np.ndarray] = {}
    for category in host["category"].unique():
        clusters = sorted(summary[summary["category"].eq(category)]["cluster"].astype(int).unique())
        for cluster in clusters:
            indices = [
                metadata_lookup[key]
                for key, value in old_lookup.items()
                if key[0] == category and value == cluster and key in metadata_lookup
            ]
            if not indices:
                raise RuntimeError(f"No centroid members for {category} cluster {cluster}")
            centroid = embeddings[indices].mean(axis=0)
            centroids[(category, cluster)] = centroid / max(np.linalg.norm(centroid), 1e-12)

    assignments = []
    assignment_values: dict[tuple[str, str], int] = {}
    reassess_keys = set(normalised_description_overrides())
    reassessment_rows = []
    for row_index, row in host.iterrows():
        key = (row["category"], normalised(row["brand_name"]))
        old_cluster = old_lookup.get(key)
        candidates = [(cid, centroid) for (category, cid), centroid in centroids.items() if category == row["category"]]
        scores = sorted(
            [(cid, float(embeddings[row_index] @ centroid)) for cid, centroid in candidates],
            key=lambda item: item[1],
            reverse=True,
        )
        best_cluster, best_score = scores[0]
        old_score = next((score for cid, score in scores if cid == old_cluster), np.nan)
        margin = best_score - old_score if old_cluster is not None else np.nan

        if key in reassess_keys:
            if old_cluster is None or (best_cluster != old_cluster and margin >= 0.025):
                cluster = best_cluster
                decision = "moved_to_better_centroid" if old_cluster is not None else "new_assignment"
            else:
                cluster = old_cluster
                decision = "kept_existing_cluster"
            if key in CLUSTER_REVIEW_OVERRIDES:
                reviewed_cluster = CLUSTER_REVIEW_OVERRIDES[key]
                if reviewed_cluster != cluster:
                    cluster = reviewed_cluster
                    decision = "manual_visual_review_override"
            reassessment_rows.append({
                "brand_name": row["brand_name"],
                "category": row["category"],
                "old_cluster": old_cluster,
                "old_cluster_title": (
                    summary_lookup[(row["category"], old_cluster)].cluster_title
                    if old_cluster is not None else ""
                ),
                "best_cluster": best_cluster,
                "best_cluster_title": summary_lookup[(row["category"], best_cluster)].cluster_title,
                "old_centroid_similarity": old_score,
                "best_centroid_similarity": best_score,
                "similarity_margin": margin,
                "adopted_cluster": cluster,
                "adopted_cluster_title": summary_lookup[(row["category"], cluster)].cluster_title,
                "decision": decision,
            })
        else:
            cluster = CLUSTER_OVERRIDES.get(key, old_cluster)
            if cluster is None:
                cluster = best_cluster
        assignment_values[key] = cluster
        info = summary_lookup[(row["category"], cluster)]
        assignments.append({
            "brand_name": row["brand_name"],
            "category": row["category"],
            "cluster": cluster,
            "cluster_id": f"{row['category']}-{cluster}",
            "cluster_title": info.cluster_title,
            "top_aesthetic_terms": info.top_aesthetic_terms,
            "supporting_terms": info.supporting_terms,
            "example_brands": info.example_brands,
            "representative_brand_name": str(info.example_brands).split(",")[0].strip(),
            "representative_moodboard_path": f"moodboards/{row['category']}/{slugify(str(info.example_brands).split(',')[0].strip())}.jpg",
            "selected_k": (
                int(info.selected_k)
                if pd.notna(info.selected_k)
                else int(summary[summary["category"].eq(row["category"])]["cluster"].nunique())
            ),
        })
    assignment_frame = pd.DataFrame(assignments)
    assignment_frame.to_csv(DATA / "ios_brand_cluster_assignments.csv", index=False)
    assignment_frame.to_csv(DATA / "ios_brand_cluster_assignments.tsv", index=False, sep="\t")
    pd.DataFrame(reassessment_rows).sort_values(
        ["decision", "similarity_margin"], ascending=[True, False]
    ).to_csv(
        reassessment_path,
        index=False,
        float_format="%.6f",
    )

    pca_values = PCA(n_components=48, random_state=42).fit_transform(embeddings)
    pca_frame = host[["brand_name", "category"]].copy()
    for index in range(48):
        pca_frame[f"e{index}"] = pca_values[:, index]
    pca_frame.to_csv(DATA / "ios_brand_embeddings_pca48.csv", index=False, float_format="%.6f")

    import umap

    umap_parts = []
    for category, category_rows in host.groupby("category", sort=False):
        indices = category_rows.index.to_numpy()
        raw = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, len(indices) - 1),
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        ).fit_transform(embeddings[indices])
        mins = raw.min(axis=0)
        spans = np.maximum(raw.max(axis=0) - mins, 1e-12)
        scaled = (raw - mins) / spans
        part = category_rows[["brand_name", "category"]].copy()
        part["cluster"] = [assignment_values[(category, normalised(name))] for name in part["brand_name"]]
        part["cluster_id"] = [f"{category}-{value}" for value in part["cluster"]]
        part["cluster_title"] = [summary_lookup[(category, int(value))].cluster_title for value in part["cluster"]]
        part["x"], part["y"] = scaled[:, 0], scaled[:, 1]
        part["umap_x"], part["umap_y"] = raw[:, 0], raw[:, 1]
        umap_parts.append(part)
    umap_frame = pd.concat(umap_parts, ignore_index=True)
    umap_frame.to_csv(DATA / "ios_brand_umap.csv", index=False, float_format="%.8f")
    umap_frame.to_csv(DATA / "ios_brand_umap.tsv", index=False, sep="\t", float_format="%.8f")

    updated_summary = summary.copy()
    for index, row in updated_summary.iterrows():
        members = assignment_frame[
            assignment_frame["category"].eq(row["category"])
            & assignment_frame["cluster"].eq(int(row["cluster"]))
        ]
        updated_summary.at[index, "n_brands"] = len(members)
    updated_summary["n_brands"] = updated_summary["n_brands"].astype(int)
    updated_summary.to_csv(DATA / "ios_cluster_summary.csv", index=False)


def copy_cluster_exports_to_ios() -> None:
    target = ROOT / "ios" / "BrandMoodboardFinder" / "BrandMoodboardFinder" / "ClusterData"
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "ios_brand_cluster_assignments.csv",
        "ios_brand_cluster_assignments.tsv",
        "ios_brand_embeddings_pca48.csv",
        "ios_brand_umap.csv",
        "ios_brand_umap.tsv",
        "ios_cluster_summary.csv",
    ):
        shutil.copy2(DATA / name, target / name)


def validate(host: pd.DataFrame, embeddings: np.ndarray) -> None:
    full_expected = len(host)
    expected = sum(
        (row.category, normalised(row.brand_name)) not in INVALID_TASTE_ROWS
        for row in host.itertuples()
    )
    files = {
        "PCA": DATA / "ios_brand_embeddings_pca48.csv",
        "clusters": DATA / "ios_brand_cluster_assignments.csv",
        "UMAP": DATA / "ios_brand_umap.csv",
    }
    if embeddings.shape != (full_expected, 768):
        raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")
    for label, path in files.items():
        rows = len(pd.read_csv(path))
        if rows != expected:
            raise RuntimeError(f"{label} has {rows} rows; expected {expected}")
    if int(pd.read_csv(DATA / "ios_cluster_summary.csv")["n_brands"].sum()) != expected:
        raise RuntimeError("Cluster summary counts do not match the integrated catalog")
    overlay = json.loads(OVERLAY.read_text(encoding="utf-8"))["entries"]
    keys = {(row.category, normalised(row.brand_name)) for row in host.itertuples()}
    missing = [entry["review_id"] for entry in overlay if (entry["category"], normalised(entry["brand_name"])) not in keys]
    if missing:
        raise RuntimeError(f"Integrated catalog is missing overlay entries: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help="Reuse the already generated 768-D matrix and rebuild only projections and cluster exports.",
    )
    args = parser.parse_args()
    host, embedding_metadata = build_catalog()
    update_source_csvs(host)
    if args.reuse_embeddings:
        embeddings = np.load(ROOT / "step_2_text_embeddings" / "brand_embeddings.npz")["embeddings"].astype("float32")
        shutil.copy2(ROOT / "step_2_text_embeddings" / "brand_embeddings.npz", DATA / "brand_embeddings.npz")
    else:
        embeddings = encode_text(embedding_metadata)
    update_projection_and_clusters(host, embeddings)
    copy_cluster_exports_to_ios()
    validate(host, embeddings)
    taste_rows = len(pd.read_csv(DATA / "ios_brand_cluster_assignments.csv"))
    print(
        f"Integrated {len(host)} brand-category rows with {embeddings.shape[1]}-D embeddings and "
        f"{taste_rows} valid taste-space rows in stable named clusters."
    )


if __name__ == "__main__":
    main()
