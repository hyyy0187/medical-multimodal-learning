"""Export study/report UID partitions without images or report text."""

from itertools import combinations

import pandas as pd

from experiment import PROJECT_ROOT, load_config, project_path


def export_splits(data_dir, output_dir):
    """Validate all partitions before writing the three public UID files."""
    partitions = {}
    for name in ("train", "val", "test"):
        frame = pd.read_csv(data_dir / f"{name}_pairs.csv", dtype={"uid": str})
        if "uid" not in frame or frame["uid"].isna().any():
            raise ValueError(f"{name} requires a nonmissing uid column.")
        partitions[name] = frame["uid"].drop_duplicates().sort_values().reset_index(drop=True)

    for first, second in combinations(partitions, 2):
        overlap = set(partitions[first]) & set(partitions[second])
        if overlap:
            raise ValueError(f"UID overlap between {first} and {second}: {len(overlap)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, uids in partitions.items():
        uids.to_frame("uid").to_csv(output_dir / f"{name}_uids.csv", index=False)
        print(f"{name}: {len(uids)} unique UIDs")
    print("UID splits are disjoint. Patient-level independence is not established.")
    return {name: len(uids) for name, uids in partitions.items()}


def main():
    cfg = load_config()
    return export_splits(project_path(cfg, "iu_xray_dir"), PROJECT_ROOT / "splits")


if __name__ == "__main__":
    main()
