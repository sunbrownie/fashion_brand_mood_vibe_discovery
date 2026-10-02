#!/usr/bin/env python3
"""Build the reversible bundled V2 catalog overlay for Ellina's added brands."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "step_3_moodboards" / "review" / "ellina_correction_manifest.json"
REVIEWS_PATH = ROOT / "moodboards_creation" / "moodboards_corrected" / "v2" / "candidate_reviews.json"
V2_MANIFEST_PATH = ROOT / "ios" / "BrandMoodboardFinder" / "BrandMoodboardFinder" / "BrandMoodboards" / "v2_manifest.json"
OUTPUT_PATH = ROOT / "ios" / "BrandMoodboardFinder" / "BrandMoodboardFinder" / "BrandMoodboards" / "v2_catalog_additions.json"
CLUSTERS_PATH = ROOT / "recommender" / "hosting_bundle" / "data" / "ios_brand_cluster_assignments.csv"


WEBSITES = {
    "A Bathing Ape": "https://bape.com", "Abercrombie & Fitch": "https://www.abercrombie.com",
    "Adidas": "https://www.adidas.com", "Arket": "https://www.arket.com", "AVAVAV": "https://avavav.com",
    "Barbour": "https://www.barbour.com", "Balmain": "https://www.balmain.com", "Bandit Running": "https://banditrunning.com",
    "Billabong": "https://www.billabong.com", "Bimba Y Lola": "https://www.bimbaylola.com", "Burton": "https://www.burton.com",
    "Burberry": "https://www.burberry.com", "Calvin Klein / Calvin Klein Jeans": "https://www.calvinklein.com",
    "Canada Goose": "https://www.canadagoose.com", "Carhartt WIP": "https://www.carhartt-wip.com",
    "Charles Jeffrey Loverboy": "https://charlesjeffreyloverboy.com", "Comme des Garcons Play": "https://www.comme-des-garcons.com",
    "Conner Ives": "https://connerives.com", "Diesel": "https://www.diesel.com", "Desigual": "https://www.desigual.com",
    "Duran Lantink": "https://duranlantink.com", "Erdem": "https://erdem.com", "GAP": "https://www.gap.com",
    "Gant": "https://www.gant.com", "Good American": "https://www.goodamerican.com", "Gymshark": "https://www.gymshark.com",
    "Gucci": "https://www.gucci.com", "Hollister": "https://www.hollisterco.com", "Jaded London": "https://jadedldn.com",
    "Jean Paul Gaultier": "https://www.jeanpaulgaultier.com", "Karl Lagerfeld": "https://www.karllagerfeld.com",
    "Lacoste": "https://www.lacoste.com", "Levi's": "https://www.levi.com", "Loro Piana": "https://www.loropiana.com",
    "Lululemon": "https://www.lululemon.com", "Mango": "https://shop.mango.com", "Martine Rose": "https://martine-rose.com",
    "Missoni": "https://www.missoni.com", "Michael Kors": "https://www.michaelkors.com", "Missguided": "https://www.missguided.com",
    "McQueen": "https://www.alexandermcqueen.com", "MKI MIYUKI ZOKU": "https://mkistore.co.uk", "Moncler": "https://www.moncler.com",
    "Moschino": "https://www.moschino.com", "Mugler": "https://fashion.mugler.com", "Napapijri": "https://www.napapijri.com",
    "Needles": "https://www.nepentheslondon.com/collections/needles", "Never Fully Dressed": "https://www.neverfullydressed.com",
    "Neighbourhood": "https://www.neighborhood.jp", "New Balance": "https://www.newbalance.com", "Nike": "https://www.nike.com",
    "No Problemo": "https://www.noproblemo.world", "Oner Active": "https://www.oneractive.com", "Open YY": "https://open-yy.com",
    "Palm Angels": "https://www.palmangels.com", "Paul Smith": "https://www.paulsmith.com",
    "Polo Ralph Lauren": "https://www.ralphlauren.com", "Puma": "https://www.puma.com", "P.E Nation": "https://www.pe-nation.com",
    "Pull & Bear": "https://www.pullandbear.com", "ABRA": "https://abra.world", "Air Jordan": "https://www.nike.com/jordan",
    "Asics": "https://www.asics.com", "Golden Goose": "https://www.goldengoose.com", "Birkenstock": "https://www.birkenstock.com",
    "EYTYS": "https://eytys.com", "Geox": "https://www.geox.com", "Gia Borghini": "https://giaborghini.com",
    "Church's": "https://www.church-footwear.com", "Crocs": "https://www.crocs.com", "Clarks": "https://www.clarks.com",
    "Ecco": "https://www.ecco.com", "Converse": "https://www.converse.com", "Carvela": "https://www.carvela.com",
    "Diemme": "https://diemme.com", "Hunter": "https://hunterboots.com", "Vivienne Westwood": "https://www.viviennewestwood.com",
    "Roxanne Assoulin": "https://roxanneassoulin.com", "Roxanne First": "https://roxannefirst.com",
    "Sonia Petroff": "https://soniapetroff.com", "Zoe Mohm": "https://zoemohm.com",
    "William Welstead": "https://williamwelstead.com", "YVMIN": "https://yvmin.com", "Yellow Swallow": "https://yellowswallow.com",
}


PROXIES = {
    "A Bathing Ape": "Kith", "Abercrombie & Fitch": "Aritzia", "Adidas": "Kappa", "Arket": "Another Aspect",
    "AVAVAV": "Ottolinger", "Barbour": "Belstaff", "Balmain": "Saint Laurent", "Bandit Running": "Arc Teryx",
    "Billabong": "Outerknown", "Bimba Y Lola": "Ganni", "Burton": "Arc Teryx", "Burberry": "Belstaff",
    "Calvin Klein / Calvin Klein Jeans": "Helmut Lang", "Canada Goose": "Mackage", "Carhartt WIP": "Dickies",
    "Charles Jeffrey Loverboy": "Chopova Lowena", "Comme des Garcons Play": "Comme des Garçons", "Conner Ives": "Marine Serre",
    "Diesel": "AG Jeans", "Desigual": "Farm Rio", "Duran Lantink": "Rick Owens", "Erdem": "Simone Rocha",
    "GAP": "Everlane", "Gant": "J.Crew", "Good American": "AG Jeans", "Gymshark": "ADANOLA",
    "Gucci": "Bottega Veneta", "Hollister": "Madewell", "Jaded London": "KNWLS", "Jean Paul Gaultier": "Coperni",
    "Karl Lagerfeld": "Akris", "Lacoste": "Fred Perry", "Levi's": "7 For All Mankind", "Loro Piana": "Brunello Cucinelli",
    "Lululemon": "ALO YOGA", "Mango": "Toteme", "Martine Rose": "A-COLD-WALL*", "Missoni": "Farm Rio",
    "Michael Kors": "J.Crew", "Missguided": "& Other Stories", "McQueen": "Ann Demeulemeester", "MKI MIYUKI ZOKU": "Aime Leon Dore",
    "Moncler": "Mackage", "Moschino": "Saint Laurent", "Mugler": "Coperni", "Napapijri": "Arc Teryx",
    "Needles": "Adsum", "Never Fully Dressed": "Farm Rio", "Neighbourhood": "Kith", "New Balance": "Kappa",
    "Nike": "Kappa", "No Problemo": "A-COLD-WALL*", "Oner Active": "ADANOLA", "Open YY": "Andersson Bell",
    "Palm Angels": "Kith", "Paul Smith": "Margaret Howell", "Polo Ralph Lauren": "J.Crew", "Puma": "Kappa",
    "P.E Nation": "ADANOLA", "Pull & Bear": "& Other Stories", "ABRA": "MIISTA", "Air Jordan": "Filling Pieces",
    "Asics": "Salomon", "Adidas|shoes": "Salomon", "Golden Goose": "Common Projects", "Birkenstock": "Arizona Love",
    "EYTYS": "Acne Studios", "Geox": "Camper", "Gia Borghini": "Amina Muaddi", "Church's": "Bruno Magli",
    "Gucci|shoes": "Bottega Veneta", "Crocs": "Calzuro", "Clarks": "Camper", "Ecco": "Camper",
    "Converse": "Common Projects", "Carvela": "Charles & Keith", "Diemme": "Fracap", "Hunter": "Blundstone",
    "Vivienne Westwood": "Ambush", "Roxanne Assoulin": "Bea Bongiasca", "Roxanne First": "Anita Ko",
    "Sonia Petroff": "Aurelie Bidermann", "Zoe Mohm": "Alighieri", "William Welstead": "Celine Daoust",
    "YVMIN": "Ambush", "Yellow Swallow": "Anni Lu",
}


LUXURY = {
    "Balmain", "Burberry", "Erdem", "Gucci", "Jean Paul Gaultier", "Loro Piana", "McQueen", "Moncler",
    "Moschino", "Mugler", "Golden Goose", "Gia Borghini", "Church's", "Vivienne Westwood", "Roxanne First",
    "Sonia Petroff", "William Welstead",
}
AFFORDABLE = {"GAP", "Hollister", "Mango", "Missguided", "Pull & Bear", "Crocs"}
UPPER = {
    "AVAVAV", "Barbour", "Bimba Y Lola", "Canada Goose", "Charles Jeffrey Loverboy", "Conner Ives", "Duran Lantink",
    "Karl Lagerfeld", "Martine Rose", "Missoni", "Michael Kors", "Needles", "Palm Angels", "Paul Smith",
    "Polo Ralph Lauren", "ABRA", "Air Jordan", "EYTYS", "Diemme", "Hunter", "Roxanne Assoulin", "Zoe Mohm", "YVMIN",
}
FEMALE_ONLY = {"Good American", "Missguided", "Never Fully Dressed", "Oner Active", "ABRA", "Gia Borghini", "Carvela",
               "Roxanne Assoulin", "Roxanne First", "Sonia Petroff", "Zoe Mohm", "William Welstead", "Yellow Swallow"}


def normalized(value: str) -> str:
    folded = "".join(c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", folded.lower().replace("&", " and ")))


def slugify(value: str) -> str:
    return normalized(value).replace(" ", "_")


def main() -> None:
    records = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["records"]
    additions = [r for r in records if r.get("recommended_action") == "catalog_add_and_generate"]
    latest_reviews = {}
    for review in json.loads(REVIEWS_PATH.read_text(encoding="utf-8"))["reviews"]:
        latest_reviews[review["review_id"]] = review
    v2_entries = json.loads(V2_MANIFEST_PATH.read_text(encoding="utf-8"))["entries"]
    v2_ids = {review_id for entry in v2_entries for review_id in entry.get("review_ids", [entry["review_id"]])}

    with CLUSTERS_PATH.open(encoding="utf-8-sig", newline="") as stream:
        cluster_rows = list(csv.DictReader(stream))
    cluster_keys = {(row["category"], normalized(row["brand_name"])) for row in cluster_rows}

    entries = []
    for record in additions:
        review_id = record["review_id"]
        name = record["brand_name"]
        category = record["category"]
        review = latest_reviews.get(review_id)
        if not review or review.get("status") != "approved":
            raise RuntimeError(f"Addition is not approved: {review_id}")
        if review_id not in v2_ids:
            raise RuntimeError(f"Addition is missing from the promoted V2 manifest: {review_id}")
        proxy = PROXIES.get(f"{name}|{category}", PROXIES.get(name))
        if not proxy or (category, normalized(proxy)) not in cluster_keys:
            raise RuntimeError(f"Invalid {category} taste proxy for {review_id}: {proxy!r}")
        website = WEBSITES.get(name)
        if not website:
            raise RuntimeError(f"Missing official website for {review_id}: {name}")
        price = "luxury" if name in LUXURY else "affordable" if name in AFFORDABLE else "upper price point" if name in UPPER else "mid price point"
        entries.append({
            "review_id": review_id,
            "brand_name": name,
            "category": category,
            "official_website": website,
            "male": name not in FEMALE_ONLY,
            "female": True,
            "price": price,
            "popularity": 1,
            "moodboard_asset_name": f"v2__{category}__{slugify(name)}.jpg",
            "taste_proxy_brand_name": proxy,
        })

    if len(entries) != 86:
        raise RuntimeError(f"Expected 86 catalog additions, found {len(entries)}")
    OUTPUT_PATH.write_text(
        json.dumps({
            "schema_version": 1,
            "policy": "Local Ellina-review V2 overlay. Remove or disable this file to roll back additions; remote catalog and V1 assets are unchanged.",
            "entries": entries,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(entries)} validated catalog additions to {OUTPUT_PATH.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
