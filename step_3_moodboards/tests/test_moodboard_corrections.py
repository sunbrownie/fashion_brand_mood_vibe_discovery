from __future__ import annotations

import json
import csv
import re
import sys
import tempfile
import unicodedata
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "step_3_moodboards" / "helpers"))
sys.path.insert(0, str(ROOT / "helpers"))

from moodboard_helpers import moodboard_prompt, save_corrected_moodboard  # noqa: E402

REVIEW_DIR = ROOT / "step_3_moodboards" / "review"
if str(REVIEW_DIR) not in sys.path:
    sys.path.insert(0, str(REVIEW_DIR))

from build_ellina_manifest import classify  # noqa: E402
from source_helpers import find_moodboard_file  # noqa: E402

STEP_DIR = ROOT / "step_3_moodboards"
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))

from build_generation_queue import (  # noqa: E402
    apply_brand_direction,
    model_safe_prompt,
    needs_generation,
    retry_prompt,
)


class MoodboardCorrectionTests(unittest.TestCase):
    def test_catalog_classification_distinguishes_board_edits_from_removals(self):
        self.assertEqual(classify("remove words form the collage. 2 of the same sweater.")[0], "targeted_regenerate")
        self.assertEqual(classify("Close vibe but remove boho bag")[0], "targeted_regenerate")
        self.assertEqual(classify("medium vibe - mostly earrings, remove rings.")[0], "targeted_regenerate")
        self.assertEqual(classify("limited jewelry, off vibe.")[0], "targeted_regenerate")
        self.assertEqual(classify("Only bags, remove from clothes")[0], "catalog_retire_or_move")
        self.assertEqual(classify("remove, don't do shoes")[0], "catalog_retire_or_move")
        self.assertEqual(classify("remove - only jewelry")[0], "catalog_retire_or_move")
        self.assertEqual(classify("Barely any bags, worth keeping?")[0], "catalog_verify")

    def test_spaced_redo_wording_is_full_regeneration(self):
        self.assertEqual(classify("R edo fully - the collage only has jewelry")[0], "full_regenerate")

    def test_retry_prompt_carries_rejection_and_category_guard(self):
        prompt = retry_prompt(
            "Base prompt.",
            "clothes",
            {"notes": "The handbag dominates the clothes board."},
            safe_output=True,
        )
        self.assertIn("clothing-only board", prompt)
        self.assertIn("handbag dominates", prompt)
        self.assertIn("fully clothed adult fashion models", prompt)
        self.assertIn("show that product only on a dress form or studio flat lay", prompt)

    def test_model_safe_prompt_removes_risky_direction(self):
        prompt = model_safe_prompt(
            "Example",
            "clothes",
            "make it sexier and sensual with explicit graphics",
        )
        self.assertIn("fully clothed adult models", prompt)
        self.assertNotIn("No people", prompt)
        self.assertNotIn("sexier", prompt.lower())
        self.assertNotIn("sensual", prompt.lower())
        self.assertNotIn("explicit", prompt.lower())

    def test_verified_brand_direction_is_applied_to_normal_generation(self):
        prompt = apply_brand_direction("Base prompt.", "jewellery", "Simon Miller")
        self.assertIn("green palm pair", prompt)
        self.assertIn("yellow banana pair", prompt)
        self.assertNotEqual(prompt, "Base prompt.")

    def test_rejected_shoe_retry_forces_one_product_per_panel(self):
        prompt = retry_prompt(
            "Base prompt.",
            "shoes",
            {"status": "rejected", "notes": "The same trainer is repeated."},
            safe_output=False,
        )
        self.assertIn("product-only editorial collage", prompt)
        self.assertIn("exactly one unique shoe or pair per panel", prompt)
        self.assertIn("Do not show people", prompt)

    def test_rejected_clothes_retry_forces_garment_led_panels(self):
        prompt = retry_prompt(
            "Base prompt.",
            "clothes",
            {"status": "rejected", "notes": "A handbag dominates the clothing board."},
            safe_output=False,
        )
        self.assertIn("exactly six panels", prompt)
        self.assertIn("hands empty", prompt)
        self.assertIn("Do not include a dedicated handbag", prompt)

    def test_repetition_note_forces_product_only_on_first_shoe_candidate(self):
        prompt = retry_prompt(
            "Base prompt.",
            "shoes",
            None,
            safe_output=False,
            reviewer_note="Close vibe, but same sandals repeated. Add more colours.",
        )
        self.assertIn("product-only editorial collage", prompt)

    def test_verified_catalog_keeps_only_supported_categories_for_regeneration(self):
        self.assertTrue(needs_generation({"review_id": "ER-046", "recommended_action": "catalog_verify"}))
        self.assertTrue(needs_generation({"review_id": "ER-244", "recommended_action": "catalog_verify"}))
        self.assertFalse(needs_generation({"review_id": "ER-234", "recommended_action": "catalog_verify"}))

    def test_acne_override_changes_generic_prompt(self) -> None:
        row = pd.Series(
            {
                "brand_name": "Acne Studios",
                "category": "clothes",
                "aesthetic_keywords": "Scandinavian, directional",
                "silhouettes": "tailoring, layered knits",
                "materials": "wool, cotton",
                "palette": "black, grey",
            }
        )
        prompt = moodboard_prompt(row, reviewer_note="Off vibe; retain the suit and rebuild the rest.")
        self.assertIn("Reviewer correction brief", prompt)
        self.assertIn("skinny-versus-layered styling contrast", prompt)
        self.assertIn("Angular cropped panels", prompt)
        self.assertNotIn("one lived-in setting", prompt)

    def test_carne_bollente_override_requires_product_and_motif_variety(self) -> None:
        row = pd.Series(
            {
                "brand_name": "Carne Bollente",
                "category": "clothes",
                "aesthetic_keywords": "playful, irreverent, graphic, cheeky",
                "silhouettes": "embroidered tees, graphic sweats, easy separates",
                "materials": "organic cotton, jersey",
                "palette": "ecru, black, brights, print",
            }
        )
        prompt = moodboard_prompt(
            row,
            reviewer_note="The brand is less about abstract prints and more about being cheeky.",
        )
        self.assertIn("six or more visibly different products", prompt)
        self.assertIn("photographic or airbrushed shirt print", prompt)
        self.assertIn("no repeated character, pose, print, embroidery, colourway, or detail crop", prompt)
        self.assertIn("repeating the same red-and-black knitted figures", prompt)

    def test_chan_luu_override_is_eclectic_without_boho(self) -> None:
        row = pd.Series(
            {
                "brand_name": "Chan Luu",
                "category": "clothes",
                "aesthetic_keywords": "eclectic, minimal, contemporary",
                "silhouettes": "dresses, tops, trousers, evening layers",
                "materials": "silk, taffeta, dupioni, mesh, sequins",
                "palette": "ivory, black, marigold, rose, navy, print",
            }
        )
        prompt = moodboard_prompt(
            row,
            reviewer_note="More eclectic and minimal-contemporary; no boho or ethnic styling.",
        )
        self.assertIn("warped plaid or graphic-check silk dress", prompt)
        self.assertIn("sequin or paillette evening layer", prompt)
        self.assertIn("unexpected feather, mesh, fringe, or shoelace construction detail", prompt)
        self.assertIn("all-neutral quiet luxury", prompt)
        self.assertIn("do not repeat plain shirts, neutral knits, or near-identical minimal looks", prompt)

    def test_second_pass_clothes_overrides_encode_requested_signatures(self) -> None:
        required_cues = {
            "Christian Wijnants": ("modern sculptural knitwear", "repeated print fragments"),
            "Destree": ("structured white high-neck poplin blouse", "Loewe-like accessories"),
            "Dime": ("solid-colour hoodie", "multiple colourways of one graphic"),
            "Girlfriend Collective": ("grounded colourful coordinated activewear sets", "neon or candy-bright colour"),
            "Hodakova": ("patchworked brown leather dress", "belts dominating more than two looks"),
            "Homecore": ("mixed casting with at least as many women as men", "men-only casting"),
            "House Of Sunny": ("playful statement knitwear", "repeating one swirl or brown-green print"),
            "Jacquemus": ("sculptural Ovalo tailoring", "bottom-only fabric close-ups"),
        }
        for brand_name, (must_include, must_avoid) in required_cues.items():
            with self.subTest(brand_name=brand_name):
                row = pd.Series(
                    {
                        "brand_name": brand_name,
                        "category": "clothes",
                        "aesthetic_keywords": "current brand direction",
                        "silhouettes": "tops, trousers, dresses, outerwear",
                        "materials": "brand-signature materials",
                        "palette": "brand-signature palette",
                    }
                )
                prompt = moodboard_prompt(row, reviewer_note="Rebuild the moodboard to match the current brand.")
                self.assertIn(must_include, prompt)
                self.assertIn(must_avoid, prompt)

    def test_third_pass_clothes_overrides_encode_requested_signatures(self) -> None:
        required_cues = {
            "Johnny Was": ("embroidered denim or suede outerwear", "sun-drenched resort styling"),
            "Karoline Vitto": ("visible sculptural metal wire bra or frame construction", "all-black palette"),
            "Kowtow": ("architectural organic-cotton dress", "anonymous beige basics"),
            "Laagam": ("green cropped jacket", "old-fashioned occasionwear"),
            "Lioness": ("horseshoe or baggy denim", "generic black tailoring collage"),
            "Lisa Says Gah": ("pastoral horse or prairie motif", "neon floral overload"),
            "Lisou": ("elegant hand-drawn silk print", "fluorescent rainbow collage"),
            "Longchamp": ("architectural Parisian outerwear", "handbag hero images"),
            "Made By Minga": ("hand-knit alpaca sweater", "candy-bright palette"),
            "Mary Katrantzou": ("Kintsugi-inspired azurite motif", "same kaleidoscope print repeated"),
            "Masscob": ("washed linen tailoring", "repeated prairie dress"),
            "Mirror Palais": ("romantic corset or bustier", "all-bridal gown collage"),
            "Miu Miu": ("playful layered tank and T-shirt", "somber grey school-uniform-only collage"),
            "MOWALOLA": ("colourful graffiti bomber", "all-black leather collage"),
            "MSGM": ("sharp colour-block tailoring", "all-over floral explosion"),
            "NILI LOTAN": ("rock-and-roll Americana tailoring", "generic corporate black suits"),
            "Nude Lucy": ("sculptural halter top", "homewear-only mood"),
            "Obey": ("solid workwear jacket", "same black-and-white graphic repeated"),
            "Opera Sport": ("asymmetric seamless top", "generic quiet-luxury layering"),
            "Patou": ("voluminous organic-cotton top", "boho floral maxi dresses"),
            "Rich Mnisi": ("sculptural compressed dress", "ethnic-costume styling"),
            "ROTATE Birger Christensen": ("peplum leather top", "all-party mini dresses"),
        }
        for brand_name, (must_include, must_avoid) in required_cues.items():
            with self.subTest(brand_name=brand_name):
                row = pd.Series(
                    {
                        "brand_name": brand_name,
                        "category": "clothes",
                        "aesthetic_keywords": "current brand direction",
                        "silhouettes": "tops, trousers, dresses, outerwear",
                        "materials": "brand-signature materials",
                        "palette": "brand-signature palette",
                    }
                )
                prompt = moodboard_prompt(row, reviewer_note="Rebuild the moodboard to match the current brand.")
                self.assertIn(must_include, prompt)
                self.assertIn(must_avoid, prompt)

    def test_corrected_file_is_preferred_only_after_approval_name_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original = root / "moodboards" / "clothes" / "acne_studios.jpg"
            candidate = root / "moodboards_corrected" / "v2" / "clothes" / "acne_studios__candidate_01.png"
            approved = root / "moodboards_corrected" / "v2" / "clothes" / "acne_studios.png"
            for path in (original, candidate):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"candidate" if "candidate" in path.name else b"original")
            self.assertEqual(find_moodboard_file(root / "moodboards", "clothes", "Acne Studios"), original)
            approved.write_bytes(b"approved")
            self.assertEqual(find_moodboard_file(root / "moodboards", "clothes", "Acne Studios"), approved)

    def test_corrected_candidate_write_is_non_destructive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = save_corrected_moodboard(
                b"image-bytes",
                "image/png",
                "bags",
                "AMI Paris",
                candidate=1,
                prompt="test prompt",
                review_id="ER-test",
                root=root,
            )
            self.assertTrue(path.exists())
            sidecar = json.loads(path.with_suffix(".png.json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["status"], "candidate_generated")
            with self.assertRaises(FileExistsError):
                save_corrected_moodboard(
                    b"new-bytes",
                    "image/png",
                    "bags",
                    "AMI Paris",
                    candidate=1,
                    prompt="test prompt",
                    review_id="ER-test",
                    root=root,
                )

    def test_all_ellina_fixtures_are_resolved(self) -> None:
        records = json.loads(
            (ROOT / "step_3_moodboards" / "review" / "ellina_correction_manifest.json").read_text(encoding="utf-8")
        )["records"]
        self.assertEqual(len(records), 360)
        self.assertEqual(sum(r["implementation_status"] == "implemented_v2" for r in records), 318)
        self.assertEqual(sum(r["implementation_status"] == "implemented_catalog_v2" for r in records), 42)
        self.assertFalse([r["review_id"] for r in records if r["implementation_status"] == "not_started"])

    def test_catalog_additions_are_complete_and_taste_enabled(self) -> None:
        asset_root = ROOT / "ios" / "BrandMoodboardFinder" / "BrandMoodboardFinder" / "BrandMoodboards"
        entries = json.loads((asset_root / "v2_catalog_additions.json").read_text(encoding="utf-8"))["entries"]
        self.assertEqual(len(entries), 86)
        self.assertEqual(len({entry["review_id"] for entry in entries}), 86)
        self.assertEqual(len({(entry["category"], entry["brand_name"]) for entry in entries}), 86)

        def normalized(value: str) -> str:
            folded = "".join(
                character
                for character in unicodedata.normalize("NFKD", value)
                if not unicodedata.combining(character)
            )
            return " ".join(re.findall(r"[a-z0-9]+", folded.lower().replace("&", " and ")))

        with (ROOT / "recommender" / "hosting_bundle" / "data" / "ios_brand_cluster_assignments.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            cluster_keys = {(row["category"], normalized(row["brand_name"])) for row in csv.DictReader(stream)}
        with (ROOT / "recommender" / "hosting_bundle" / "data" / "ios_brand_embeddings_pca48.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            embedding_keys = {(row["category"], normalized(row["brand_name"])) for row in csv.DictReader(stream)}

        for entry in entries:
            self.assertTrue(entry["official_website"].startswith("https://"), entry["review_id"])
            self.assertTrue((asset_root / entry["moodboard_asset_name"]).exists(), entry["review_id"])
            proxy_key = (entry["category"], normalized(entry["taste_proxy_brand_name"]))
            self.assertIn(proxy_key, cluster_keys, entry["review_id"])
            self.assertIn(proxy_key, embedding_keys, entry["review_id"])

    def test_v2_manifest_uses_app_compatible_slugs_and_preserves_rollback(self) -> None:
        asset_root = ROOT / "ios" / "BrandMoodboardFinder" / "BrandMoodboardFinder" / "BrandMoodboards"
        entries = json.loads((asset_root / "v2_manifest.json").read_text(encoding="utf-8"))["entries"]
        self.assertEqual(len(entries), 318)
        self.assertEqual(len({entry["ios_v2_path"] for entry in entries}), 318)

        def slug(value: str) -> str:
            folded = "".join(
                character
                for character in unicodedata.normalize("NFKD", value)
                if not unicodedata.combining(character)
            )
            return "_".join(re.findall(r"[a-z0-9]+", folded.lower().replace("&", " and ")))

        for entry in entries:
            expected_name = f"v2__{entry['category']}__{slug(entry['brand_name'])}.jpg"
            self.assertEqual(Path(entry["ios_v2_path"]).name, expected_name, entry["review_id"])
            self.assertTrue((ROOT / "ios" / "BrandMoodboardFinder" / entry["ios_v2_path"]).exists())
            self.assertIn("V1 asset is unchanged", entry["rollback"])

    def test_review_reference_links_and_screenshots_cover_every_fixture(self) -> None:
        payload = json.loads((REVIEW_DIR / "reference_sources.json").read_text(encoding="utf-8"))
        records = payload["records"]
        self.assertEqual(len(records), 360)
        self.assertEqual(len({record["review_id"] for record in records}), 360)
        unique_screenshots = {record["screenshot_path"] for record in records}
        self.assertEqual(len(unique_screenshots), 359)
        for record in records:
            self.assertTrue(record["official_url"].startswith("https://"), record["review_id"])
            self.assertIn("google.com/search?", record["google_images_url"], record["review_id"])
        for relative_path in unique_screenshots:
            screenshot = ROOT / relative_path
            self.assertTrue(screenshot.exists(), relative_path)
            with Image.open(screenshot) as image:
                self.assertEqual(image.size, (600, 900), relative_path)


if __name__ == "__main__":
    unittest.main()
