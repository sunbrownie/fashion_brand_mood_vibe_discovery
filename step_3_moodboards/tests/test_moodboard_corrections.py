from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "step_3_moodboards" / "helpers"))
sys.path.insert(0, str(ROOT / "helpers"))

from moodboard_helpers import moodboard_prompt, save_corrected_moodboard  # noqa: E402
from source_helpers import find_moodboard_file  # noqa: E402


class MoodboardCorrectionTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
