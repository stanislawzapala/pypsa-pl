import csv
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INSTALLED_CAPACITY_FILE = (
    REPOSITORY_ROOT
    / "data/input/installed_capacity;variant=historical+instrat_projection.csv"
)


def load_installed_capacity_rows():
    with INSTALLED_CAPACITY_FILE.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


class InstalledCapacityDataTest(unittest.TestCase):
    def test_cumulative_rows_are_one_year_snapshots(self):
        rows = [
            row
            for row in load_installed_capacity_rows()
            if row["cumulative"].strip().upper() == "TRUE"
        ]

        self.assertTrue(rows)
        self.assertTrue(all(row["build_year"] == row["retire_year"] for row in rows))

    def test_projection_contains_national_snapshots_through_2050(self):
        rows = load_installed_capacity_rows()
        years = {int(row["build_year"]) for row in rows if row["build_year"]}
        areas = {row["area"] for row in rows if row["area"]}

        self.assertTrue({2025, 2030, 2035, 2040, 2045, 2050}.issubset(years))
        self.assertEqual(max(years), 2050)
        self.assertEqual(areas, {"PL"})

    def test_main_generation_and_storage_technologies_are_aggregated_to_pl(self):
        rows = load_installed_capacity_rows()
        technologies = [
            "hard coal power old",
            "hard coal power SC",
            "lignite power old",
            "lignite power SC",
            "natural gas power CCGT",
            "natural gas power peaker",
            "wind onshore",
            "wind offshore",
            "solar PV ground",
            "solar PV roof",
            "nuclear power large",
            "battery large power",
            "battery large storage",
        ]

        for technology in technologies:
            with self.subTest(technology=technology):
                technology_rows = [
                    row for row in rows if row["technology"] == technology
                ]

                self.assertTrue(technology_rows)
                self.assertEqual({row["area"] for row in technology_rows}, {"PL"})


if __name__ == "__main__":
    unittest.main()
