
# %% Part 1: Load Day 6 fine-tuned model

import hashlib
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoTokenizer, AutoModel

from experiment import (
    dataloader_options, get_device, load_config,
    project_path, seed_everything, save_config,
)

try:
    from IPython.display import display
except ImportError:
    display = print

cfg = load_config()
DATA_DIR = project_path(cfg, "iu_xray_dir")
RESULT_DIR = project_path(cfg, "results_dir")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED = cfg["seed"]
seed_everything(SEED, deterministic=cfg["runtime"]["deterministic"])
DEVICE = get_device(cfg)

IMAGE_BATCH_SIZE = cfg["robustness"]["image_batch_size"]
TEXT_BATCH_SIZE = cfg["robustness"]["text_batch_size"]
CLEAN_MAX_LENGTH = cfg["text"]["max_length"]

BERT_NAME = cfg["models"]["text_name"]


checkpoint = torch.load(
    RESULT_DIR / "best_partial_finetune_fixed.pth",
    map_location="cpu",
    weights_only=True
)


class InferenceMedicalCLIP(nn.Module):

    def __init__(self, projection_dim):
        super().__init__()

        self.image_encoder = resnet18(weights=None)
        self.image_encoder.fc = nn.Identity()

        self.image_projection = nn.Linear(
            512, projection_dim
        )

        self.text_projection = nn.Linear(
            768, projection_dim
        )

    def encode_image(self, images):

        features = self.image_encoder(images)
        z = self.image_projection(features)

        return F.normalize(z, dim=-1)

    def encode_text(self, features):

        z = self.text_projection(features)

        return F.normalize(z, dim=-1)


state = checkpoint["model_state_dict"]

projection_dim = (
    state["image_projection.weight"].shape[0]
)
cfg["models"]["projection_dim"] = projection_dim
cfg["loaded_checkpoint"] = {
    "path": str(RESULT_DIR / "best_partial_finetune_fixed.pth"),
    "projection_dim": projection_dim,
    "projection_dim_source": "checkpoint weights",
}
save_config(cfg, RESULT_DIR / "robustness_config.yaml")

model = InferenceMedicalCLIP(
    projection_dim
).to(DEVICE)

model.load_state_dict(state)
model.eval()

print("Device:", DEVICE)
print("Best fine-tuning epoch:", checkpoint["epoch"])
print("Projection dimension:", projection_dim)


# %% Part 2: Load validation pairs

val_df = pd.read_csv(
    DATA_DIR / "val_pairs.csv",
    dtype={"uid": str, "filename": str}
)

val_df = (
    val_df.sort_values(["uid", "filename"])
    .drop_duplicates("uid")
    .reset_index(drop=True)
)

# Recover original report text if needed
if "report" not in val_df.columns:

    manifest = pd.read_csv(
        DATA_DIR / "matched_pairs_with_features.csv",
        dtype={"uid": str, "filename": str}
    )

    report_lookup = (
        manifest[["uid", "filename", "report"]]
        .drop_duplicates(["uid", "filename"])
    )

    val_df = val_df.merge(
        report_lookup,
        on=["uid", "filename"],
        how="left",
        validate="one_to_one"
    )

assert val_df["report"].notna().all()

REPORTS = val_df["report"].astype(str).tolist()
N = len(val_df)

print("Validation pairs:", N)


# Load Day 3 cached BERT features
text_data = np.load(
    DATA_DIR / "iu_bert_report_features.npz"
)

text_features = text_data["features"].astype(np.float32)
text_uids = text_data["report_uids"].astype(str)

text_rows = val_df["feature_row"].astype(int).to_numpy()

assert np.array_equal(
    text_uids[text_rows],
    val_df["uid"].to_numpy()
)

X_txt = torch.from_numpy(
    text_features[text_rows].copy()
)

print("Text feature shape:", X_txt.shape)


# %% Part 3: Retrieval metrics

def retrieval_metrics(image_z, text_z):

    similarity = (
        image_z @ text_z.T
    ).detach().cpu().numpy()

    n = similarity.shape[0]
    assert similarity.shape == (n, n)

    results = []

    for direction, scores in [
        ("I2T", similarity),
        ("T2I", similarity.T)
    ]:

        order = np.argsort(
            -scores,
            axis=1,
            kind="stable"
        )

        ranks = (
            np.argmax(
                order == np.arange(n)[:, None],
                axis=1
            ) + 1
        )

        results.append({
            "Direction": direction,
            "R@1": 100 * np.mean(ranks <= 1),
            "R@5": 100 * np.mean(ranks <= 5),
            "R@10": 100 * np.mean(ranks <= 10),
            "MedR": float(np.median(ranks))
        })

    return results, similarity


@torch.no_grad()
def project_text_features(features):

    outputs = []

    for start in range(0, len(features), 128):

        batch = features[
            start:start + 128
        ].to(DEVICE)

        outputs.append(
            model.encode_text(batch).cpu()
        )

    return torch.cat(outputs, dim=0)


clean_text_z = project_text_features(X_txt)


# %% Part 4: Image perturbation dataset

IMAGE_DIR = (
    DATA_DIR / "images" / "images_normalized"
)


def image_path_of(row):

    rel = row.get("image_path")

    if pd.notna(rel):
        path = DATA_DIR / str(rel)

        if path.is_file():
            return path

    path = IMAGE_DIR / str(row["filename"])

    if path.is_file():
        return path

    raise FileNotFoundError(row["filename"])


class ImagePerturbDataset(Dataset):

    def __init__(self, df, sigma=0.0):

        self.paths = [
            image_path_of(row)
            for row in df.to_dict("records")
        ]

        self.sigma = sigma
        self.transform = (
            ResNet18_Weights.DEFAULT.transforms()
        )

        self.mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)

        self.std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        with Image.open(self.paths[idx]) as img:
            x = self.transform(img.convert("RGB"))

        if self.sigma > 0:

            generator = torch.Generator()
            generator.manual_seed(SEED + idx)

            noise = torch.randn(
                x.shape,
                generator=generator
            ) * self.sigma

            x = x + noise

            # Clamp to the normalized RGB value range
            min_value = -self.mean / self.std
            max_value = (1 - self.mean) / self.std

            x = torch.maximum(
                torch.minimum(x, max_value),
                min_value
            )

        return x


@torch.no_grad()
def encode_images(sigma=0.0):

    dataset = ImagePerturbDataset(
        val_df,
        sigma=sigma
    )

    loader = DataLoader(
        dataset,
        batch_size=IMAGE_BATCH_SIZE,
        shuffle=False,
        **dataloader_options(cfg),
    )

    outputs = []

    for images in loader:

        images = images.to(DEVICE)

        image_z = model.encode_image(images)

        outputs.append(image_z.cpu())

    return torch.cat(outputs, dim=0)


# %% Part 5: Verify clean baseline

clean_image_z = encode_images(sigma=0.0)

clean_results, clean_similarity = (
    retrieval_metrics(
        clean_image_z,
        clean_text_z
    )
)

print("Clean baseline:")
print(
    pd.DataFrame(clean_results)
    .round(3)
    .to_string(index=False)
)

mean_r5 = np.mean(
    [row["R@5"] for row in clean_results]
)

print("Mean Recall@5:", round(mean_r5, 3))

if abs(mean_r5 - float(checkpoint["mean_r5"])) > 0.1:
    print(
        "WARNING: Baseline differs from Day 6. "
        "Check model, validation order and preprocessing."
    )


# %% Part 6: Image Gaussian noise experiment

robustness_records = []


def record_metrics(family, severity, image_z, text_z):

    results, _ = retrieval_metrics(
        image_z, text_z
    )

    for row in results:
        robustness_records.append({
            "Family": family,
            "Severity": severity,
            **row
        })

    print(
        family,
        severity,
        "| I2T R@5:",
        round(results[0]["R@5"], 3),
        "| T2I R@5:",
        round(results[1]["R@5"], 3)
    )


NOISE_LEVELS = cfg["robustness"]["image_noise_levels"]

for sigma in NOISE_LEVELS:

    if sigma == 0:
        image_z = clean_image_z
    else:
        image_z = encode_images(sigma=sigma)

    record_metrics(
        "image_gaussian",
        sigma,
        image_z,
        clean_text_z
    )


# %% Part 7: Reload BioClinicalBERT

tokenizer = AutoTokenizer.from_pretrained(
    BERT_NAME
)

text_encoder = AutoModel.from_pretrained(
    BERT_NAME
).to(DEVICE)

text_encoder.eval()


@torch.no_grad()
def encode_reports(texts, max_length=CLEAN_MAX_LENGTH):

    features = []

    for start in range(
        0, len(texts), TEXT_BATCH_SIZE
    ):

        batch_texts = texts[
            start:start + TEXT_BATCH_SIZE
        ]

        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(DEVICE)

        outputs = text_encoder(**tokens)

        hidden = outputs.last_hidden_state

        mask = (
            tokens["attention_mask"]
            .unsqueeze(-1)
            .to(hidden.dtype)
        )

        pooled = (
            (hidden * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )

        features.append(pooled.cpu())

    return torch.cat(features, dim=0)


# %% Part 8: Verify text feature consistency

recomputed_features = encode_reports(
    REPORTS,
    max_length=CLEAN_MAX_LENGTH
)

cosine_similarity = F.cosine_similarity(
    F.normalize(X_txt, dim=1),
    F.normalize(recomputed_features, dim=1),
    dim=1
)

print(
    "Mean feature cosine:",
    cosine_similarity.mean().item()
)

recomputed_text_z = project_text_features(
    recomputed_features
)

recomputed_results, _ = retrieval_metrics(
    clean_image_z,
    recomputed_text_z
)

print("\nRe-encoded clean text baseline:")
print(
    pd.DataFrame(recomputed_results)
    .round(3)
    .to_string(index=False)
)

if cosine_similarity.mean().item() < 0.999:
    print(
        "\nWARNING: Re-encoded features differ "
        "from Day 3 cached features."
    )
    print(
        "Check original report text, "
        "max_length and pooling method."
    )



# %% Part 9: Word dropout robustness

PROTECT_WORDS = {
    "no", "not", "without", "never",
    "neither", "nor", "negative",
    "absent", "absence", "denies"
}


def drop_words(text, rate, uid):

    words = str(text).split()

    # Stable random seed based on UID
    uid_hash = hashlib.sha1(
        str(uid).encode("utf-8")
    ).hexdigest()

    uid_seed = int(uid_hash[:8], 16)

    rng = np.random.default_rng(
        SEED + uid_seed
    )

    kept = []

    for word in words:

        normalized = re.sub(
            r"^[^a-zA-Z]+|[^a-zA-Z]+$",
            "",
            word
        ).lower()

        if normalized in PROTECT_WORDS:
            kept.append(word)

        elif rng.random() >= rate:
            kept.append(word)

    return " ".join(kept) or str(text)


WORD_DROPOUT_RATES = cfg["robustness"]["word_dropout_rates"]

for rate in WORD_DROPOUT_RATES:

    if rate == 0:
        text_z = recomputed_text_z

    else:
        noisy_reports = [
            drop_words(text, rate, uid)
            for text, uid in zip(
                REPORTS,
                val_df["uid"]
            )
        ]

        noisy_features = encode_reports(
            noisy_reports,
            max_length=CLEAN_MAX_LENGTH
        )

        text_z = project_text_features(
            noisy_features
        )

    record_metrics(
        "text_word_dropout",
        rate,
        clean_image_z,
        text_z
    )


# %% Part 10: Text truncation experiment

MAX_LENGTHS = cfg["robustness"]["max_lengths"]

for max_length in MAX_LENGTHS:

    if max_length == CLEAN_MAX_LENGTH:
        text_z = recomputed_text_z

    else:
        truncated_features = encode_reports(
            REPORTS,
            max_length=max_length
        )

        text_z = project_text_features(
            truncated_features
        )

    record_metrics(
        "text_truncation",
        max_length,
        clean_image_z,
        text_z
    )


# %% Part 11: Save robustness results

robustness_df = pd.DataFrame(
    robustness_records
)

output_path = (
    RESULT_DIR / "day7_robustness.csv"
)

robustness_df.to_csv(
    output_path,
    index=False
)

print(
    robustness_df.round(3)
    .to_string(index=False)
)

print("\nSaved:", output_path)


# %% Part 12: Plot robustness curves

def plot_robustness(
    family,
    xlabel,
    title,
    filename
):

    subset = robustness_df[
        robustness_df["Family"] == family
    ]

    plt.figure(figsize=(6.5, 4))

    for direction in ["I2T", "T2I"]:

        df = subset[
            subset["Direction"] == direction
        ].sort_values("Severity")

        plt.plot(
            df["Severity"],
            df["R@5"],
            marker="o",
            label=direction
        )

    plt.xlabel(xlabel)
    plt.ylabel("Recall@5 (%)")
    plt.title(title)

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / filename,
        dpi=200
    )

    plt.show()


plot_robustness(
    "image_gaussian",
    "Gaussian noise sigma",
    "Image Robustness",
    "day7_image_robustness.png"
)

plot_robustness(
    "text_word_dropout",
    "Word dropout rate",
    "Text Noise Robustness",
    "day7_text_noise_robustness.png"
)

plot_robustness(
    "text_truncation",
    "Maximum token length",
    "Text Truncation Robustness",
    "day7_text_truncation.png"
)


# %% Part 13: Layer4 ablation

# Backup the fine-tuned layer4 parameters
original_layer4 = {
    name: tensor.detach().cpu().clone()
    for name, tensor in
    model.image_encoder.layer4.state_dict().items()
}

# Load original ImageNet-pretrained ResNet18
reference_resnet = resnet18(
    weights=ResNet18_Weights.DEFAULT
)

# Replace fine-tuned layer4
model.image_encoder.layer4.load_state_dict(
    reference_resnet.layer4.state_dict()
)

model.eval()

# Evaluate after resetting layer4
reset_image_z = encode_images(sigma=0.0)

reset_results, _ = retrieval_metrics(
    reset_image_z,
    clean_text_z
)

print("Reset layer4:")
print(
    pd.DataFrame(reset_results)
    .round(3)
    .to_string(index=False)
)

# Restore fine-tuned layer4
model.image_encoder.layer4.load_state_dict(
    original_layer4
)

model.eval()

del reference_resnet

print("\nFine-tuned layer4 restored.")


# %% Part 14: Ablation comparison

ablation_rows = []

for method, results in [
    ("Fine-tuned layer4", clean_results),
    ("Reset layer4", reset_results)
]:

    for row in results:
        ablation_rows.append({
            "Method": method,
            **row
        })


# Add Day 6 frozen-encoder baseline
comparison_path = (
    RESULT_DIR / "retrieval_comparison.csv"
)

if comparison_path.exists():

    previous = pd.read_csv(comparison_path)

    frozen_rows = previous[
        previous["Method"] == "Trained Dual Encoder"
    ]

    for row in frozen_rows.to_dict("records"):
        ablation_rows.append({
            "Method": "Frozen Encoder",
            "Direction": row["Direction"],
            "R@1": row["R@1"],
            "R@5": row["R@5"],
            "R@10": row["R@10"],
            "MedR": row["MedR"]
        })


ablation_df = pd.DataFrame(
    ablation_rows
)

ablation_df.to_csv(
    RESULT_DIR / "day7_ablation.csv",
    index=False
)

print(
    ablation_df.round(3)
    .to_string(index=False)
)


# %% Part 15: Retrieval failure analysis

def normalize_text(text):
    return re.sub(
        r"\s+",
        " ",
        str(text).strip().lower()
    )


failure_rows = []

for direction, scores in [
    ("I2T", clean_similarity),
    ("T2I", clean_similarity.T)
]:

    rankings = np.argsort(
        -scores,
        axis=1,
        kind="stable"
    )

    for i in range(N):

        top1_idx = int(rankings[i, 0])

        # Skip correct Top-1 predictions
        if top1_idx == i:
            continue

        correct_rank = int(
            np.flatnonzero(
                rankings[i] == i
            )[0] + 1
        )

        query = val_df.iloc[i]
        predicted = val_df.iloc[top1_idx]

        failure_rows.append({
            "Direction": direction,
            "query_uid": query["uid"],
            "top1_uid": predicted["uid"],
            "correct_rank": correct_rank,
            "correct_similarity": float(
                scores[i, i]
            ),
            "top1_similarity": float(
                scores[i, top1_idx]
            ),
            "query_filename": query["filename"],
            "top1_filename": predicted["filename"],
            "correct_report": query["report"],
            "top1_report": predicted["report"],
            "same_report_text": (
                normalize_text(query["report"])
                == normalize_text(predicted["report"])
            )
        })


failure_df = pd.DataFrame(
    failure_rows
)

failure_df = failure_df.sort_values(
    "correct_rank",
    ascending=False
)

failure_df.to_csv(
    RESULT_DIR / "day7_failure_cases.csv",
    index=False
)

print("Total failure cases:", len(failure_df))

display(
    failure_df[
        [
            "Direction",
            "query_uid",
            "top1_uid",
            "correct_rank",
            "same_report_text"
        ]
    ].head(10)
)


# %% Part 16: Inspect a failed retrieval

case = failure_df[
    failure_df["Direction"] == "I2T"
].iloc[0]

print("Query UID:", case["query_uid"])
print("Top-1 predicted UID:", case["top1_uid"])
print("Correct rank:", case["correct_rank"])

print("\n=== Correct report ===")
print(case["correct_report"])

print("\n=== Retrieved report ===")
print(case["top1_report"])

print("\n=== Similarity ===")
print("Correct:", case["correct_similarity"])
print("Top-1:", case["top1_similarity"])


# Display original chest X-ray
row = val_df[
    val_df["uid"] == case["query_uid"]
].iloc[0]

image_path = image_path_of(row)

with Image.open(image_path) as img:

    plt.figure(figsize=(5, 5))
    plt.imshow(img.convert("L"), cmap="gray")
    plt.title(
        f"Failed I2T retrieval | UID {case['query_uid']}"
    )
    plt.axis("off")
    plt.show()


# %% Part 18: Seen vs Unseen report retrieval

# Load train reports
train_data = pd.read_csv(
    DATA_DIR / "train_pairs.csv",
    dtype={"uid": str, "filename": str}
)

if "report" not in train_data.columns:

    manifest = pd.read_csv(
        DATA_DIR / "matched_pairs_with_features.csv",
        dtype={"uid": str, "filename": str}
    )

    lookup = (
        manifest[["uid", "filename", "report"]]
        .drop_duplicates(["uid", "filename"])
    )

    train_data = train_data.merge(
        lookup,
        on=["uid", "filename"],
        how="left",
        validate="many_to_one"
    )


def normalize_report(text):
    return re.sub(
        r"\s+",
        " ",
        str(text).lower().strip()
    )


train_texts = set(
    train_data["report"]
    .dropna()
    .map(normalize_report)
)

# val_df must match clean_similarity row order
seen_mask = (
    val_df["report"]
    .map(normalize_report)
    .isin(train_texts)
    .to_numpy()
)

print("Seen:", seen_mask.sum())
print("Unseen:", (~seen_mask).sum())


def get_ranks(scores):

    n = scores.shape[0]

    order = np.argsort(
        -scores,
        axis=1,
        kind="stable"
    )

    return (
        np.argmax(
            order == np.arange(n)[:, None],
            axis=1
        ) + 1
    )


rows = []

for direction, scores in [
    ("I2T", clean_similarity),
    ("T2I", clean_similarity.T)
]:

    ranks = get_ranks(scores)

    for group_name, mask in [
        ("Seen Reports", seen_mask),
        ("Unseen Reports", ~seen_mask)
    ]:

        group_ranks = ranks[mask]

        rows.append({
            "Group": group_name,
            "Direction": direction,
            "N": len(group_ranks),
            "R@1": np.mean(group_ranks <= 1) * 100,
            "R@5": np.mean(group_ranks <= 5) * 100,
            "R@10": np.mean(group_ranks <= 10) * 100,
            "MedR": np.median(group_ranks)
        })


seen_unseen_df = pd.DataFrame(rows)

display(seen_unseen_df.round(3))

seen_unseen_df.to_csv(
    RESULT_DIR / "day7_seen_unseen_comparison.csv",
    index=False
)


# %% Part 19: Duplicate reports within validation set

val_normalized = (
    val_df["report"]
    .map(normalize_report)
)

# True if the same text occurs for multiple UIDs
val_duplicate_mask = val_normalized.duplicated(
    keep=False
)

group_labels = np.where(
    seen_mask,
    "Seen",
    "Unseen"
)

summary = pd.DataFrame({
    "Group": group_labels,
    "Duplicate_in_validation": val_duplicate_mask
})

print("Validation duplicate report analysis:")

print(
    pd.crosstab(
        summary["Group"],
        summary["Duplicate_in_validation"],
        margins=True
    )
)

print(
    "\nUnique report texts:",
    val_normalized.nunique()
)

print(
    "Validation records with duplicated text:",
    int(val_duplicate_mask.sum())
)

# %%
