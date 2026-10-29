
# %% Part 1: Load dataset splits

import pandas as pd
import numpy as np
import hashlib
import re

from experiment import load_config, project_path

try:
    from IPython.display import display
except ImportError:
    display = print

cfg = load_config()
DATA_DIR = project_path(cfg, "iu_xray_dir")
RESULT_DIR = project_path(cfg, "results_dir")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def load_split(name):
    df = pd.read_csv(
        DATA_DIR / f"{name}_pairs.csv",
        dtype={"uid": str, "filename": str}
    )
    return df


train_df = load_split("train")
val_df = load_split("val")
test_df = load_split("test")

splits = {
    "train": train_df,
    "val": val_df,
    "test": test_df
}

for name, df in splits.items():
    print(
        name,
        "Records:", len(df),
        "Unique UID:", df["uid"].nunique(),
        "Unique images:", df["filename"].nunique()
    )

print("\nAvailable columns:")
print(train_df.columns.tolist())


# %% Part 2: UID and filename overlap

split_pairs = [
    ("train", "val"),
    ("train", "test"),
    ("val", "test")
]

audit_rows = []

for name_a, name_b in split_pairs:

    df_a = splits[name_a]
    df_b = splits[name_b]

    uids_a = set(df_a["uid"])
    uids_b = set(df_b["uid"])

    files_a = set(df_a["filename"])
    files_b = set(df_b["filename"])

    uid_overlap = uids_a & uids_b
    file_overlap = files_a & files_b

    audit_rows.append({
        "Split": f"{name_a}/{name_b}",
        "UID_overlap": len(uid_overlap),
        "Filename_overlap": len(file_overlap)
    })

audit_df = pd.DataFrame(audit_rows)

print(audit_df.to_string(index=False))

assert (audit_df["UID_overlap"] == 0).all()
assert (audit_df["Filename_overlap"] == 0).all()

print("\nUID and filename overlap checks passed.")


# %% Part 3: Patient-level overlap

patient_candidates = [
    "patient_id",
    "patientid",
    "subject_id",
    "patient",
    "pid"
]

patient_col = next(
    (
        col for col in patient_candidates
        if all(col in df.columns for df in splits.values())
    ),
    None
)

if patient_col is None:
    print("No patient ID column found.")
    print("Patient-level overlap cannot be verified.")

    audit_df["Patient_overlap"] = "UNVERIFIABLE"

else:
    print("Candidate patient ID column:", patient_col)

    print(
        "IMPORTANT: Confirm this column represents "
        "a real patient identifier."
    )

    for i, (name_a, name_b) in enumerate(split_pairs):

        a = set(
            splits[name_a][patient_col]
            .dropna().astype(str).str.strip()
        )

        b = set(
            splits[name_b][patient_col]
            .dropna().astype(str).str.strip()
        )

        a.discard("")
        b.discard("")

        audit_df.loc[i, "Patient_overlap"] = len(a & b)

print(audit_df.to_string(index=False))

# %% Part 4: Exact report text overlap

def normalize_report(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Restore report text if it is not stored in split CSVs
if not all(
    "report" in df.columns
    for df in splits.values()
):
    manifest = pd.read_csv(
        DATA_DIR / "matched_pairs_with_features.csv",
        dtype={"uid": str, "filename": str}
    )

    report_lookup = (
        manifest[["uid", "filename", "report"]]
        .drop_duplicates(["uid", "filename"])
    )

    for name, df in splits.items():
        if "report" not in df.columns:
            splits[name] = df.merge(
                report_lookup,
                on=["uid", "filename"],
                how="left",
                validate="many_to_one"
            )


report_sets = {}

for name, df in splits.items():
    normalized = df["report"].map(normalize_report)

    report_sets[name] = set(
        text for text in normalized
        if text != ""
    )


for i, (name_a, name_b) in enumerate(split_pairs):

    overlap = (
        report_sets[name_a]
        & report_sets[name_b]
    )

    audit_df.loc[
        i, "Exact_report_overlap"
    ] = len(overlap)

print(audit_df.to_string(index=False))


# %% Part 5: Image hash overlap

IMAGE_DIR = (
    DATA_DIR / "images" / "images_normalized"
)

def find_image(row):

    # Prefer the saved path when available
    if "image_path" in row:
        value = row["image_path"]

        if pd.notna(value):
            candidate = DATA_DIR / str(value)

            if candidate.is_file():
                return candidate

    # Fallback to the normalized image directory
    candidate = IMAGE_DIR / str(row["filename"])

    if candidate.is_file():
        return candidate

    return None


def file_sha256(path):

    hasher = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            hasher.update(chunk)

    return hasher.hexdigest()


hash_sets = {}
hash_cache = {}

for name, df in splits.items():

    hashes = set()
    missing = 0

    for row in df.to_dict("records"):

        path = find_image(row)

        if path is None:
            missing += 1
            continue

        if path not in hash_cache:
            hash_cache[path] = file_sha256(path)

        hashes.add(hash_cache[path])

    hash_sets[name] = hashes

    print(
        name,
        "Unique hashes:", len(hashes),
        "Missing images:", missing
    )


for i, (name_a, name_b) in enumerate(split_pairs):

    overlap = (
        hash_sets[name_a]
        & hash_sets[name_b]
    )

    audit_df.loc[
        i, "Image_hash_overlap"
    ] = len(overlap)

print(audit_df.to_string(index=False))


# %% Part 6: Save audit results

output_path = (
    RESULT_DIR / "day7_data_audit.csv"
)

audit_df.to_csv(
    output_path,
    index=False
)

print("\nFinal data audit:")
print(audit_df.to_string(index=False))

print("\nSaved:", output_path)


# %% Fix checkpoint serialization

import torch

path = RESULT_DIR / "best_partial_finetune.pth"

# Only load with weights_only=False because
# this is our own trusted checkpoint.
checkpoint = torch.load(
    path,
    map_location="cpu",
    weights_only=False
)

# Check the original data type
print("Original type:", type(checkpoint["mean_r5"]))

# Convert NumPy scalar to Python float
checkpoint["mean_r5"] = float(checkpoint["mean_r5"])
checkpoint["epoch"] = int(checkpoint["epoch"])
checkpoint["pilot"] = bool(checkpoint["pilot"])

# Save a separate, cleaned checkpoint
fixed_path = path.with_name(
    "best_partial_finetune_fixed.pth"
)

torch.save(checkpoint, fixed_path)

# Verify safe loading
verified = torch.load(
    fixed_path,
    map_location="cpu",
    weights_only=True
)

print("Fixed checkpoint loaded successfully.")
print("Epoch:", verified["epoch"])
print("Mean R@5:", verified["mean_r5"])


# %% Part 7: Validation report duplication rate

train_texts = set(
    splits["train"]["report"]
    .map(normalize_report)
)

val_unique = (
    splits["val"]
    .sort_values(["uid", "filename"])
    .drop_duplicates("uid")
    .copy()
)

val_unique["normalized_report"] = (
    val_unique["report"].map(normalize_report)
)

val_unique["seen_in_train"] = (
    val_unique["normalized_report"].isin(train_texts)
)

n_seen = int(val_unique["seen_in_train"].sum())
n_total = len(val_unique)

print("Validation UID count:", n_total)
print("Reports seen in training:", n_seen)
print(
    "Percentage:",
    round(n_seen / n_total * 100, 2),
    "%"
)

print("\nExamples:")
display(
    val_unique.loc[
        val_unique["seen_in_train"],
        ["uid", "report"]
    ].head(5)
)

# %%
