
# %% Part 1: Prepare validation data

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from experiment import (
    load_config, project_path, seed_everything, get_device, save_config,
)

cfg = load_config()
DATA_DIR = project_path(cfg, "iu_xray_dir")
RESULT_DIR = project_path(cfg, "results_dir")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED = cfg["seed"]
seed_everything(SEED, deterministic=cfg["runtime"]["deterministic"])
device = get_device(cfg)

# Load pretrained features
img_data = np.load(
    DATA_DIR / "iu_image_features.npz"
)
txt_data = np.load(
    DATA_DIR / "iu_bert_report_features.npz"
)

image_features = img_data["features"].astype(np.float32)
text_features = txt_data["features"].astype(np.float32)

image_names = img_data["filenames"].astype(str)
image_uids = img_data["report_uids"].astype(str)
text_uids = txt_data["report_uids"].astype(str)

image_index = {
    name: i for i, name in enumerate(image_names)
}

# Same validation data as Day 5
val_df = pd.read_csv(
    DATA_DIR / "val_pairs.csv",
    dtype={"uid": str, "filename": str}
)

# One image per UID
val_df = (
    val_df.sort_values(["uid", "filename"])
    .drop_duplicates(subset="uid")
    .reset_index(drop=True)
)

img_rows = []
txt_rows = []

for row in val_df.itertuples(index=False):

    i = image_index[row.filename]
    j = int(row.feature_row)

    assert image_uids[i] == row.uid
    assert text_uids[j] == row.uid

    img_rows.append(i)
    txt_rows.append(j)

X_img = torch.from_numpy(
    image_features[img_rows].copy()
).to(device)

X_txt = torch.from_numpy(
    text_features[txt_rows].copy()
).to(device)

N = len(val_df)

assert N == len(set(val_df["uid"]))

print("Device:", device)
print("Validation pairs:", N)
print("Image features:", X_img.shape)
print("Text features:", X_txt.shape)


# %% Part 2: Retrieval metrics

def calculate_ranks(similarity):
    """
    similarity: [N, N]
    Correct matches are on the diagonal.
    """

    n = similarity.shape[0]

    assert similarity.shape == (n, n)

    # Sort candidates by descending similarity
    order = np.argsort(
        -similarity,
        axis=1,
        kind="stable"
    )

    # Convert indices into 1-based ranks
    ranks = np.empty_like(order)

    ranks[
        np.arange(n)[:, None],
        order
    ] = np.arange(1, n + 1)[None, :]

    return ranks[np.arange(n), np.arange(n)]


def evaluate_retrieval(similarity, method):

    results = []

    for direction, scores in [
        ("I2T", similarity),
        ("T2I", similarity.T)
    ]:

        ranks = calculate_ranks(scores)

        results.append({
            "Method": method,
            "Direction": direction,
            "R@1": np.mean(ranks <= 1) * 100,
            "R@5": np.mean(ranks <= 5) * 100,
            "R@10": np.mean(ranks <= 10) * 100,
            "MedR": np.median(ranks)
        })

    return results


# %% Part 3: Random retrieval baseline

rng = np.random.default_rng(SEED)

random_results = []

# Average over the configured number of random retrieval runs.
for repeat in range(cfg["retrieval"]["random_repeats"]):

    # Random scores independent of true matching
    random_similarity = rng.random((N, N))

    random_results.extend(
        evaluate_retrieval(
            random_similarity,
            "Random Retrieval"
        )
    )

random_df = pd.DataFrame(random_results)

random_summary = (
    random_df.groupby(
        ["Method", "Direction"],
        as_index=False
    )[["R@1", "R@5", "R@10", "MedR"]]
    .mean()
)

print("Random retrieval baseline:")
print(random_summary.round(3))

print("\nTheoretical chance recall:")
for k in [1, 5, 10]:
    print(f"R@{k}: {min(k, N)/N*100:.3f}%")


# %% Part 4: Untrained projection baseline

class ProjectionCLIP(nn.Module):

    def __init__(self, projection_dim=256):
        super().__init__()

        self.image_projection = nn.Linear(
            512, projection_dim
        )

        self.text_projection = nn.Linear(
            768, projection_dim
        )

    def forward(self, img, txt):

        image_z = self.image_projection(img)
        text_z = self.text_projection(txt)

        image_z = F.normalize(image_z, dim=-1)
        text_z = F.normalize(text_z, dim=-1)

        return image_z, text_z


CHECKPOINT_PATH = (
    RESULT_DIR / "best_projection_model.pth"
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True
)

# Checkpoint architecture takes precedence when evaluating an existing model.
projection_dim = checkpoint.get(
    "projection_dim", cfg["models"]["projection_dim"]
)
# Record the effective architecture rather than the YAML's requested dimension.
cfg["models"]["projection_dim"] = projection_dim
cfg["loaded_checkpoint"] = {
    "path": str(CHECKPOINT_PATH),
    "projection_dim": projection_dim,
    "projection_dim_source": (
        "checkpoint" if "projection_dim" in checkpoint else "config.yaml"
    ),
}

# Randomly initialized projection heads
seed_everything(SEED, deterministic=cfg["runtime"]["deterministic"])

untrained_model = ProjectionCLIP(
    projection_dim=projection_dim
).to(device)

untrained_model.eval()

with torch.no_grad():

    image_z, text_z = untrained_model(X_img, X_txt)

    untrained_similarity = (
        image_z @ text_z.T
    ).cpu().numpy()

untrained_results = evaluate_retrieval(
    untrained_similarity,
    "Untrained Dual Encoder"
)

print(pd.DataFrame(untrained_results).round(3))


# %% Part 5: Trained dual encoder

trained_model = ProjectionCLIP(
    projection_dim=projection_dim
).to(device)

trained_model.load_state_dict(
    checkpoint["model_state_dict"]
)

trained_model.eval()

print("Loaded best epoch:", checkpoint["epoch"])

with torch.no_grad():

    image_z, text_z = trained_model(X_img, X_txt)

    trained_similarity = (
        image_z @ text_z.T
    ).cpu().numpy()

trained_results = evaluate_retrieval(
    trained_similarity,
    "Trained Dual Encoder"
)

print(pd.DataFrame(trained_results).round(3))


# %% Part 6: Compare three baselines

comparison = pd.concat(
    [
        random_summary,
        pd.DataFrame(untrained_results),
        pd.DataFrame(trained_results)
    ],
    ignore_index=True
)

print("\n=== Retrieval Comparison ===")
print(comparison.round(3).to_string(index=False))

# Save results
save_config(
    cfg,
    RESULT_DIR / "retrieval_config.yaml",
)
comparison.to_csv(
    RESULT_DIR / "retrieval_comparison.csv",
    index=False
)

# Plot Recall@1 comparison
labels = (
    comparison["Method"] +
    " (" + comparison["Direction"] + ")"
)

plt.figure(figsize=(10, 5))

plt.bar(
    labels,
    comparison["R@1"]
)

plt.ylabel("Recall@1 (%)")
plt.title("Image-Text Retrieval Baseline Comparison")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "retrieval_comparison.png",
    dpi=200
)

plt.show()

print("Comparison table and figure saved.")
