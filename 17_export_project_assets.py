"""Validate aggregate result tables and export the curated README figures.

This script only handles the explicit allowlists below. It never exports images
from the dataset, feature arrays, checkpoints, or report-level failure examples.
"""

import csv
import math
from pathlib import Path
import shutil

from experiment import PROJECT_ROOT, load_config, project_path


FIGURES = (
    "training_curve.png",
    "retrieval_comparison.png",
    "day7_image_robustness.png",
    "day7_text_noise_robustness.png",
    "day7_text_truncation.png",
)

RESULT_TABLES = {
    "training_history.csv": {"epoch", "train_loss", "val_loss"},
    "retrieval_comparison.csv": {"Method", "Direction", "R@1", "R@5", "R@10", "MedR"},
    "day6_final_comparison.csv": {"Method", "Direction", "R@1", "R@5", "R@10", "MedR"},
    "day7_robustness.csv": {"Family", "Severity", "Direction", "R@1", "R@5", "R@10", "MedR"},
    "day7_ablation.csv": {"Method", "Direction", "R@1", "R@5", "R@10", "MedR"},
    "day7_data_audit.csv": {"Split", "UID_overlap", "Filename_overlap", "Patient_overlap", "Exact_report_overlap", "Image_hash_overlap"},
    "day7_seen_unseen_comparison.csv": {"Group", "Direction", "N", "R@1", "R@5", "R@10", "MedR"},
}

SUMMARY_NAME = "day6_final_comparison.csv"
RETRIEVAL_COLUMNS = ("Method", "Direction", "R@1", "R@5", "R@10", "MedR")


def comparison_rows(tables):
    """Build the four-row summary only when both frozen baselines agree."""
    def select(name, method):
        selected = [row for row in tables[name] if row["Method"] == method]
        if len(selected) != 2 or {row["Direction"] for row in selected} != {"I2T", "T2I"}:
            raise ValueError(f"{name} requires one I2T and one T2I row for {method}.")
        for row in selected:
            for metric in RETRIEVAL_COLUMNS[2:]:
                value = float(row[metric])
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite {metric} for {method} in {name}.")
        return {row["Direction"]: row for row in selected}

    frozen = select("retrieval_comparison.csv", "Trained Dual Encoder")
    audited_frozen = select("day7_ablation.csv", "Frozen Encoder")
    fine_tuned = select("day7_ablation.csv", "Fine-tuned layer4")
    for direction in ("I2T", "T2I"):
        for metric in RETRIEVAL_COLUMNS[2:]:
            if not math.isclose(
                float(frozen[direction][metric]),
                float(audited_frozen[direction][metric]),
                rel_tol=1e-10, abs_tol=1e-10,
            ):
                raise ValueError(
                    f"Frozen baseline differs between retrieval_comparison.csv and "
                    f"day7_ablation.csv ({direction}, {metric}); rerun related evaluations."
                )

    return [
        {**frozen[direction], "Method": "Frozen Encoder"}
        for direction in ("I2T", "T2I")
    ] + [fine_tuned[direction] for direction in ("I2T", "T2I")]


def export_assets(result_dir, figure_dir):
    """Check every input before copying figures; keep tables in results/."""
    result_dir, figure_dir = Path(result_dir), Path(figure_dir)
    input_tables = {name: columns for name, columns in RESULT_TABLES.items() if name != SUMMARY_NAME}
    required = [result_dir / name for name in (*FIGURES, *input_tables)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing experiment artifacts:\n" + "\n".join(missing))

    tables = {}
    for name, expected in input_tables.items():
        with (result_dir / name).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            if columns != expected or len(reader.fieldnames) != len(expected):
                raise ValueError(f"Unexpected columns in {name}: {sorted(columns)}")
            rows = list(reader)
            if not rows:
                raise ValueError(f"Empty result table: {name}")
            tables[name] = rows
        print(f"Validated results/{name}")

    for name in FIGURES:
        with (result_dir / name).open("rb") as handle:
            if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                raise ValueError(f"Not a PNG image: {name}")

    summary = comparison_rows(tables)
    # The summary is derived, so a missing or stale copy is replaced only after
    # every source table, figure, and cross-table consistency check has passed.
    with (result_dir / SUMMARY_NAME).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RETRIEVAL_COLUMNS)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Updated results/{SUMMARY_NAME}")

    figure_dir.mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        source, target = result_dir / name, figure_dir / name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        print(f"Exported figures/{name}")
    return {"tables": len(RESULT_TABLES), "figures": len(FIGURES)}


def main():
    cfg = load_config()
    return export_assets(project_path(cfg, "results_dir"), PROJECT_ROOT / "figures")


if __name__ == "__main__":
    main()
