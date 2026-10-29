
# %% Part 1: Match real reports with image metadata

import re
import pandas as pd
from experiment import (
    load_config, seed_everything, get_device, project_path, dataloader_options,
)

cfg = load_config()
seed_everything(cfg["seed"], deterministic=cfg["runtime"]["deterministic"])
DATA_DIR = project_path(cfg, "iu_xray_dir")

reports = pd.read_csv(DATA_DIR / "indiana_reports.csv")
projections = pd.read_csv(DATA_DIR / "indiana_projections.csv")

print("Reports:", reports.shape)
print("Projections:", projections.shape)

# Combine Findings and Impression
def clean_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()

reports["report"] = (
    reports["findings"].fillna("") + " " +
    reports["impression"].fillna("")
).map(clean_text)

# Remove empty reports
reports = reports[reports["report"] != ""].copy()

# Use frontal images only
projections = projections[
    projections["projection"].str.lower() == "frontal"
].copy()

# Match using report ID
pairs = projections.merge(
    reports[["uid", "report"]],
    on="uid",
    how="inner"
)

pairs = pairs[["uid", "filename", "report"]]
pairs = pairs.dropna(subset=["filename"])
pairs = pairs.drop_duplicates()

print("Matched pairs:", len(pairs))
print("Unique reports:", pairs["uid"].nunique())

print("\nFirst matched sample:")
print(pairs.iloc[0].to_dict())

# Save matching metadata
pairs.to_csv(
    DATA_DIR / "matched_reports.csv",
    index=False
)


# %% Part 2: Text Dataset and DataLoader

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

MODEL_NAME = cfg["models"]["text_name"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ReportDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        return {
            "uid": str(row["uid"]),
            "filename": row["filename"],
            "text": row["report"]
        }

def collate_fn(batch):
    texts = [item["text"] for item in batch]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=cfg["text"]["max_length"],
        return_tensors="pt"
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "uid": [item["uid"] for item in batch],
        "filename": [item["filename"] for item in batch]
    }

# First test 32 pairs
dataset = ReportDataset(pairs.head(32))

loader = DataLoader(
    dataset,
    batch_size=cfg["text"]["batch_size"],
    shuffle=False,
    **dataloader_options(cfg),
    collate_fn=collate_fn
)

batch = next(iter(loader))

print("Input IDs:", batch["input_ids"].shape)
print("Attention Mask:", batch["attention_mask"].shape)
print("Report IDs:", batch["uid"][:3])
print("Image filenames:", batch["filename"][:3])



# %% Part 3: Extract BERT features

import numpy as np
from transformers import AutoModel

# 每份报告只编码一次
unique_reports = pairs.drop_duplicates(
    subset="uid"
).reset_index(drop=True)

print("Unique reports:", len(unique_reports))

# 检查 Token 长度
tokenized = tokenizer(
    unique_reports["report"].tolist(),
    padding=False,
    truncation=False
)

lengths = np.array([
    len(ids) for ids in tokenized["input_ids"]
])

print("Median token length:", np.median(lengths))
print("95th percentile:", np.percentile(lengths, 95))
max_length = cfg["text"]["max_length"]
print(f"Truncated at {max_length}:", (lengths > max_length).sum())

# 加载预训练 BERT
device = get_device(cfg)

bert = AutoModel.from_pretrained(
    MODEL_NAME
).to(device)

bert.eval()

# 创建 DataLoader
text_dataset = ReportDataset(unique_reports)

text_loader = DataLoader(
    text_dataset,
    batch_size=cfg["text"]["batch_size"],
    shuffle=False,
    **dataloader_options(cfg),
    collate_fn=collate_fn
)

feature_batches = []
report_uids = []

with torch.inference_mode():

    for i, batch in enumerate(text_loader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden = outputs.last_hidden_state

        # Masked Mean Pooling
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)

        features = (
            (hidden * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )

        feature_batches.append(features.cpu())
        report_uids.extend(batch["uid"])

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(text_loader)} batches")

all_features = torch.cat(
    feature_batches, dim=0
).numpy()

print("Final feature matrix:", all_features.shape)


# %% Part 4: Save features and matching metadata

# 保存文本特征
np.savez_compressed(
    DATA_DIR / "iu_bert_report_features.npz",
    report_uids=np.array(report_uids),
    features=all_features
)

# 建立 uid 到特征矩阵行号的对应关系
uid_to_row = {
    str(uid): i
    for i, uid in enumerate(report_uids)
}

matched = pairs.copy()
matched["uid"] = matched["uid"].astype(str)

matched["feature_row"] = (
    matched["uid"].map(uid_to_row)
)

# 确保所有图像都能找到报告特征
assert matched["feature_row"].notna().all()

matched["feature_row"] = (
    matched["feature_row"].astype(int)
)

matched.to_csv(
    DATA_DIR / "matched_pairs_with_features.csv",
    index=False
)

print("Number of image-report pairs:", len(matched))
print("Feature dimensions:", all_features.shape)
print("Missing features:", matched["feature_row"].isna().sum())
print("All files saved successfully!")


# %%
# 读取 Day 3 保存的文本特征
data = np.load(
    DATA_DIR / "iu_bert_report_features.npz"
)

# 查看文件中包含哪些数组
print("Keys:", data.files)

for key in data.files:
    arr = data[key]
    print(key, arr.shape, arr.dtype)

features = data["features"]
uids = data["report_uids"]

print("First report UID:", uids[8])
print("Feature shape:", features[0].shape)
print("First 10 values:", features[0, :10])
