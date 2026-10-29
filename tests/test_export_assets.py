"""Boundary checks for publishing only explicitly curated project artifacts."""

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SPLITS = load_script("16_export_splits.py")
ASSETS = load_script("17_export_project_assets.py")


class ExportTests(unittest.TestCase):
    def setUp(self):
        context = redirect_stdout(io.StringIO())
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def create_assets(self, source):
        source.mkdir()
        for name, columns in ASSETS.RESULT_TABLES.items():
            if name == ASSETS.SUMMARY_NAME:
                continue
            pd.DataFrame({column: ["fixture"] for column in sorted(columns)}).to_csv(
                source / name, index=False,
            )

        def rows(method, recall):
            return [
                {"Method": method, "Direction": direction,
                 "R@1": 1.0, "R@5": recall, "R@10": 12.0, "MedR": 90.0}
                for direction in ("I2T", "T2I")
            ]

        pd.DataFrame(rows("Trained Dual Encoder", 5.0)).to_csv(
            source / "retrieval_comparison.csv", index=False,
        )
        pd.DataFrame(rows("Frozen Encoder", 5.0) + rows("Fine-tuned layer4", 7.0)).to_csv(
            source / "day7_ablation.csv", index=False,
        )
        for name in ASSETS.FIGURES:
            (source / name).write_bytes(b"\x89PNG\r\n\x1a\nfixture")

    def test_uid_export_excludes_report_text_and_sorts_unique_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "local", root / "published"
            source.mkdir()
            for name, ids in {"train": ["b", "a", "a"], "val": ["c"], "test": ["d"]}.items():
                pd.DataFrame({"uid": ids, "report": ["local only"] * len(ids)}).to_csv(
                    source / f"{name}_pairs.csv", index=False,
                )
            counts = SPLITS.export_splits(source, target)
            self.assertEqual(counts, {"train": 2, "val": 1, "test": 1})
            result = pd.read_csv(target / "train_uids.csv")
            self.assertEqual(list(result.columns), ["uid"])
            self.assertEqual(list(result["uid"]), ["a", "b"])

    def test_overlapping_uids_fail_before_writing_any_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("train", "val", "test"):
                pd.DataFrame({"uid": ["same"]}).to_csv(root / f"{name}_pairs.csv", index=False)
            output = root / "published"
            with self.assertRaisesRegex(ValueError, "overlap"):
                SPLITS.export_splits(root, output)
            self.assertFalse(output.exists())

    def test_missing_asset_fails_before_creating_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                ASSETS.export_assets(root, root / "published")
            self.assertFalse((root / "published").exists())

    def test_asset_export_is_repeatable_and_does_not_copy_private_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "local", root / "published"
            self.create_assets(source)
            (source / "day7_failure_cases.csv").write_text("private report text", encoding="utf-8")
            (source / "model.pth").write_bytes(b"weights")
            (source / "chest_xray.png").write_bytes(b"raw image")
            (source / "features.npz").write_bytes(b"cached features")
            (source / "data" / "nested").mkdir(parents=True)
            (source / "data" / "nested" / "raw.png").write_bytes(b"raw image")
            expected = {"tables": 7, "figures": 5}
            self.assertEqual(ASSETS.export_assets(source, target), expected)
            self.assertEqual(ASSETS.export_assets(source, target), expected)
            self.assertEqual({path.name for path in target.iterdir()}, set(ASSETS.FIGURES))
            summary = pd.read_csv(source / ASSETS.SUMMARY_NAME)
            self.assertEqual(list(summary["Method"]), [
                "Frozen Encoder", "Frozen Encoder", "Fine-tuned layer4", "Fine-tuned layer4",
            ])

    def test_changed_source_metrics_update_the_derived_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "local", root / "published"
            self.create_assets(source)
            ASSETS.export_assets(source, target)
            frame = pd.read_csv(source / "day7_ablation.csv")
            frame.loc[frame["Method"] == "Fine-tuned layer4", "R@5"] = 8.5
            frame.to_csv(source / "day7_ablation.csv", index=False)
            ASSETS.export_assets(source, target)
            summary = pd.read_csv(source / ASSETS.SUMMARY_NAME)
            self.assertEqual(list(summary["R@5"]), [5.0, 5.0, 8.5, 8.5])

    def test_mismatched_baselines_fail_without_replacing_the_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "local", root / "published"
            self.create_assets(source)
            summary = source / ASSETS.SUMMARY_NAME
            summary.write_text("previous result", encoding="utf-8")
            frame = pd.read_csv(source / "day7_ablation.csv")
            frame.loc[frame["Method"] == "Frozen Encoder", "R@5"] = 6.0
            frame.to_csv(source / "day7_ablation.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Frozen baseline differs"):
                ASSETS.export_assets(source, target)
            self.assertEqual(summary.read_text(encoding="utf-8"), "previous result")
            self.assertFalse(target.exists())

    def test_report_text_columns_are_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "local", root / "published"
            self.create_assets(source)
            frame = pd.read_csv(source / "retrieval_comparison.csv")
            frame["report"] = "private report text"
            frame.to_csv(source / "retrieval_comparison.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Unexpected columns"):
                ASSETS.export_assets(source, target)
            self.assertFalse((source / ASSETS.SUMMARY_NAME).exists())
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
