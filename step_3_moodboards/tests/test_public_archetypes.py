import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "recommender" / "hosting_bundle" / "data"


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalized(value: str) -> str:
    return " ".join("".join(character.lower() if character.isalnum() else " " for character in value).split())


class PublicArchetypeArtifactTests(unittest.TestCase):
    def test_canonical_archetypes_have_unique_complete_customer_copy(self):
        archetypes = rows("public_cluster_archetypes.csv")
        self.assertEqual(len(archetypes), 15)
        self.assertEqual(len({row["public_cluster_id"] for row in archetypes}), 15)
        self.assertEqual(len({row["public_cluster_name"] for row in archetypes}), 15)
        for row in archetypes:
            self.assertTrue(row["public_cluster_name"].strip())
            self.assertTrue(row["public_cluster_description"].strip())

    def test_every_taste_assignment_has_consistent_public_archetype(self):
        archetypes = {row["public_cluster_id"]: row for row in rows("public_cluster_archetypes.csv")}
        assignments = rows("ios_brand_cluster_assignments.csv")
        self.assertEqual(len(assignments), 1799)
        self.assertEqual(
            len({(row["brand_name"], row["category"]) for row in assignments}),
            len(assignments),
        )
        for row in assignments:
            canonical = archetypes[row["public_cluster_id"]]
            self.assertEqual(row["public_cluster_name"], canonical["public_cluster_name"])
            self.assertEqual(row["public_cluster_description"], canonical["public_cluster_description"])
            self.assertTrue(row["public_display_name"].strip())
            self.assertTrue(row["public_display_description"].strip())
            self.assertTrue(row["cluster_id"].startswith(f"{row['category']}-"))

    def test_reviewed_split_cluster_anchors(self):
        assignments = {
            (row["category"], normalized(row["brand_name"])): row
            for row in rows("ios_brand_cluster_assignments.csv")
        }
        expected = {
            ("clothes", "alo yoga"): "Endorphin",
            ("clothes", "marine layer"): "Sun Bleached",
            ("clothes", "pucci"): "The Trove",
            ("clothes", "carhartt wip"): "The Workshop",
            ("clothes", "a cold wall"): "After Midnight",
        }
        for key, public_name in expected.items():
            if key in assignments:
                self.assertEqual(assignments[key]["public_cluster_name"], public_name)

    def test_umap_and_assignment_public_fields_match_without_changing_technical_ids(self):
        assignments = {
            (row["category"], row["brand_name"]): row
            for row in rows("ios_brand_cluster_assignments.csv")
        }
        umap = rows("ios_brand_umap.csv")
        self.assertEqual(len(umap), len(assignments))
        for row in umap:
            assignment = assignments[(row["category"], row["brand_name"])]
            self.assertEqual(row["cluster"], assignment["cluster"])
            self.assertEqual(row["cluster_id"], assignment["cluster_id"])
            self.assertEqual(row["public_cluster_id"], assignment["public_cluster_id"])
            self.assertEqual(row["public_cluster_name"], assignment["public_cluster_name"])
            self.assertEqual(row["public_display_name"], assignment["public_display_name"])
            self.assertEqual(row["public_display_description"], assignment["public_display_description"])

    def test_public_summary_reconciles_to_assignments(self):
        assignments = rows("ios_brand_cluster_assignments.csv")
        public_summary = rows("ios_public_cluster_summary.csv")
        self.assertEqual(sum(int(row["n_brands"]) for row in public_summary), len(assignments))
        self.assertEqual(
            {(row["category"], row["public_cluster_id"]) for row in public_summary},
            {(row["category"], row["public_cluster_id"]) for row in assignments},
        )


if __name__ == "__main__":
    unittest.main()
