# %% Part 1: Verify image-report matching

import pandas as pd
import numpy as np
from experiment import (
    load_config, seed_everything, get_device, project_path, dataloader_options,
)

cfg = load_config()
seed_everything(cfg["seed"], deterministic=cfg["runtime"]["deterministic"])
DATA_DIR = project_path(cfg, "iu_xray_dir")
IMAGE_DIR = DATA_DIR / "images"

# Load previously matched metadata
pairs = pd.read_csv(
    DATA_DIR / "matched_pairs_with_features.csv",
    dtype={"uid": str, "filename": str}
)

# Load previously extracted BERT features
saved = np.load(
    DATA_DIR / "iu_bert_report_features.npz"
)

text_features = saved["features"]
report_uids = saved["report_uids"].astype(str)

print("Original pairs:", len(pairs))
print("Text features:", text_features.shape)

# Find actual image files
if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(
        f"Image folder not found: {IMAGE_DIR}"
    )

image_files = sorted(IMAGE_DIR.rglob("*.png"))

print("Image files found:", len(image_files))

# Build filename -> relative path mapping
image_map = {}

for path in image_files:
    if path.name in image_map:
        raise ValueError(
            f"Duplicate image filename: {path.name}"
        )

    image_map[path.name] = str(
        path.relative_to(DATA_DIR)
    )

# Match image filenames with actual paths
pairs["image_path"] = pairs["filename"].map(image_map)

missing = pairs["image_path"].isna().sum()

print("Missing images:", missing)

if missing > 0:
    print(
        "Missing examples:",
        pairs.loc[
            pairs["image_path"].isna(), "filename"
        ].head(5).tolist()
    )

pairs = pairs.dropna(
    subset=["image_path"]
).reset_index(drop=True)

if pairs.empty:
    raise ValueError("No valid image-text pairs found.")

# Verify text feature indices
pairs["feature_row"] = pd.to_numeric(
    pairs["feature_row"],
    errors="raise"
).astype(int)

assert (pairs["feature_row"] >= 0).all()
assert (pairs["feature_row"] < len(report_uids)).all()

assert np.array_equal(
    report_uids[pairs["feature_row"].to_numpy()],
    pairs["uid"].to_numpy()
)

print("Valid image-text pairs:", len(pairs))
print("Unique report IDs:", pairs["uid"].nunique())
print("Text feature matching: PASSED")


# %% Part 2: Split dataset by UID

from sklearn.model_selection import GroupShuffleSplit

# First split: hold out the configured fraction by report UID
splitter1 = GroupShuffleSplit(
    n_splits=1,
    test_size=cfg["pairs"]["holdout_fraction"],
    random_state=cfg["seed"]
)

train_idx, temp_idx = next(
    splitter1.split(
        pairs,
        groups=pairs["uid"]
    )
)

train_df = pairs.iloc[train_idx].copy()
temp_df = pairs.iloc[temp_idx].copy()

# Second split: temporary set into validation and test
splitter2 = GroupShuffleSplit(
    n_splits=1,
    test_size=cfg["pairs"]["test_fraction_of_holdout"],
    random_state=cfg["seed"]
)

val_idx, test_idx = next(
    splitter2.split(
        temp_df,
        groups=temp_df["uid"]
    )
)

val_df = temp_df.iloc[val_idx].copy()
test_df = temp_df.iloc[test_idx].copy()

# Ensure no UID overlap
train_ids = set(train_df["uid"])
val_ids = set(val_df["uid"])
test_ids = set(test_df["uid"])

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

print("\nDataset splitting:")
print("Train pairs:", len(train_df))
print("Validation pairs:", len(val_df))
print("Test pairs:", len(test_df))
print("UID overlap: 0")

# Save split metadata
train_df.to_csv(
    DATA_DIR / "train_pairs.csv",
    index=False
)

val_df.to_csv(
    DATA_DIR / "val_pairs.csv",
    index=False
)

test_df.to_csv(
    DATA_DIR / "test_pairs.csv",
    index=False
)

print("Dataset splits saved successfully!")


# %% Part 3: Build Image-Text Dataset and DataLoader

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from torchvision.models import ResNet18_Weights
from transformers import AutoTokenizer

MODEL_NAME = cfg["models"]["text_name"]

# Text tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

# Image preprocessing
weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()


class ImageTextDataset(Dataset):

    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = DATA_DIR / row["image_path"]

        # Read and preprocess image
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        image = self.transform(image)

        return {
            "image": image,
            "text": row["report"],
            "uid": str(row["uid"]),
            "filename": row["filename"],
            "feature_row": int(row["feature_row"])
        }


def collate_image_text(batch):

    # Stack images into a batch
    images = torch.stack([
        item["image"] for item in batch
    ])

    # Collect texts
    texts = [
        item["text"] for item in batch
    ]

    # Tokenize clinical reports
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=cfg["text"]["max_length"],
        return_tensors="pt"
    )

    return {
        "image": images,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "uid": [item["uid"] for item in batch],
        "filename": [
            item["filename"] for item in batch
        ],
        "feature_row": torch.tensor(
            [item["feature_row"] for item in batch],
            dtype=torch.long
        )
    }


# Create train Dataset
train_dataset = ImageTextDataset(
    train_df,
    transform=preprocess
)

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg["pairs"]["batch_size"],
    shuffle=True,
    **dataloader_options(cfg),
    collate_fn=collate_image_text
)

# Test one batch
batch = next(iter(train_loader))

print("\nMultimodal DataLoader test:")
print("Image:", batch["image"].shape)
print("Input IDs:", batch["input_ids"].shape)
print("Attention Mask:", batch["attention_mask"].shape)
print("UID:", batch["uid"][:3])
print("Feature rows:", batch["feature_row"][:3])


# %% Part 4: Extract ResNet-18 image features

import torch.nn as nn
from torchvision.models import resnet18

device = get_device(cfg)

print("\nDevice:", device)

# Load pretrained image encoder
model = resnet18(weights=weights)

# Remove classification layer
model.fc = nn.Identity()

model = model.to(device)
model.eval()

# Create Dataset for all valid image-text pairs
export_dataset = ImageTextDataset(
    pairs,
    transform=preprocess
)

export_loader = DataLoader(
    export_dataset,
    batch_size=cfg["pairs"]["export_batch_size"],
    shuffle=False,
    **dataloader_options(cfg),
    collate_fn=collate_image_text
)

image_feature_batches = []
export_filenames = []
export_uids = []

with torch.inference_mode():

    for i, batch in enumerate(export_loader):

        images = batch["image"].to(device)

        features = model(images)

        image_feature_batches.append(
            features.cpu().numpy()
        )

        export_filenames.extend(batch["filename"])
        export_uids.extend(batch["uid"])

        if (i + 1) % 50 == 0:
            print(
                f"Processed {i+1}/{len(export_loader)} batches"
            )

# Combine all image features
image_features = np.concatenate(
    image_feature_batches,
    axis=0
)

# Verify sample order
assert export_filenames == pairs["filename"].tolist()
assert export_uids == pairs["uid"].tolist()

print("\nFinal results:")
print("Image feature matrix:", image_features.shape)
print("Text feature matrix:", text_features.shape)

# Save image features
np.savez_compressed(
    DATA_DIR / "iu_image_features.npz",
    filenames=np.array(export_filenames),
    report_uids=np.array(export_uids),
    features=image_features
)

# Save paired metadata
pairs.to_csv(
    DATA_DIR / "paired_manifest.csv",
    index=False
)

print("Image features saved successfully!")
print("Paired manifest saved successfully!")
