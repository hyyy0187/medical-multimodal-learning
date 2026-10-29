
# %% Part 1: Load paired features

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

from losses import clip_loss
from experiment import (
    load_config, project_path, seed_everything, get_device,
    dataloader_options, save_config,
)

cfg = load_config()
training_cfg = cfg["contrastive"]
DATA_DIR = project_path(cfg, "iu_xray_dir")
RESULT_DIR = project_path(cfg, "results_dir")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
SEED = cfg["seed"]
seed_everything(SEED, deterministic=cfg["runtime"]["deterministic"])

device = get_device(cfg)

print("Device:", device)

# Load features
image_data = np.load(
    DATA_DIR / "iu_image_features.npz"
)

text_data = np.load(
    DATA_DIR / "iu_bert_report_features.npz"
)

image_features = image_data["features"].astype(np.float32)
text_features = text_data["features"].astype(np.float32)

image_names = image_data["filenames"].astype(str)
image_uids = image_data["report_uids"].astype(str)
text_uids = text_data["report_uids"].astype(str)

# Filename -> feature matrix row
assert len(set(image_names)) == len(image_names)

image_index = {
    name: idx for idx, name in enumerate(image_names)
}

print("Image features:", image_features.shape)
print("Text features:", text_features.shape)


# %% Part 2: Prepare unique paired features

train_df = pd.read_csv(
    DATA_DIR / "train_pairs.csv",
    dtype={"uid": str, "filename": str}
)

val_df = pd.read_csv(
    DATA_DIR / "val_pairs.csv",
    dtype={"uid": str, "filename": str}
)

# One image per report for the baseline
def select_unique_reports(df):
    return (
        df.sort_values(["uid", "filename"])
          .drop_duplicates(subset="uid")
          .reset_index(drop=True)
    )

train_df = select_unique_reports(train_df)
val_df = select_unique_reports(val_df)

assert set(train_df["uid"]).isdisjoint(set(val_df["uid"]))

def build_feature_dataset(df):
    img_rows = []
    txt_rows = []

    for row in df.itertuples(index=False):
        image_row = image_index[row.filename]
        text_row = int(row.feature_row)

        # Verify image and report identity
        assert image_uids[image_row] == row.uid
        assert text_uids[text_row] == row.uid

        img_rows.append(image_row)
        txt_rows.append(text_row)

    X_img = torch.from_numpy(
        image_features[img_rows].copy()
    )

    X_txt = torch.from_numpy(
        text_features[txt_rows].copy()
    )

    return TensorDataset(X_img, X_txt)

train_dataset = build_feature_dataset(train_df)
val_dataset = build_feature_dataset(val_df)

print("Train unique pairs:", len(train_dataset))
print("Validation unique pairs:", len(val_dataset))

print("Example image shape:", train_dataset[0][0].shape)
print("Example text shape:", train_dataset[0][1].shape)

# %% Part 3: DataLoader and projection model

BATCH_SIZE = training_cfg["batch_size"]
PROJECTION_DIM = cfg["models"]["projection_dim"]
TEMPERATURE = training_cfg["temperature"]
LEARNING_RATE = training_cfg["learning_rate"]
WEIGHT_DECAY = training_cfg["weight_decay"]
EPOCHS = training_cfg["epochs"]
PATIENCE = training_cfg["patience"]

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=training_cfg["train_drop_last"],
    **dataloader_options(cfg),
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=training_cfg["val_drop_last"],
    **dataloader_options(cfg),
    pin_memory=(device.type == "cuda")
)

# This is the trainable projection portion
# of the frozen-encoder MedicalCLIP model.
class ProjectionCLIP(nn.Module):

    def __init__(self, projection_dim=256):
        super().__init__()

        self.image_projection = nn.Linear(
            512, projection_dim
        )

        self.text_projection = nn.Linear(
            768, projection_dim
        )

    def forward(self, image_features, text_features):

        image_z = self.image_projection(image_features)
        text_z = self.text_projection(text_features)

        image_z = F.normalize(image_z, dim=-1)
        text_z = F.normalize(text_z, dim=-1)

        return image_z, text_z


model = ProjectionCLIP(
    projection_dim=PROJECTION_DIM
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters())
)


# %% Part 4: Training and validation functions

def train_one_epoch(model, loader, optimizer):
    model.train()

    total_loss = 0.0

    for image_x, text_x in loader:

        image_x = image_x.to(device)
        text_x = text_x.to(device)

        optimizer.zero_grad()

        # Forward
        image_z, text_z = model(image_x, text_x)

        # Contrastive loss
        loss = clip_loss(
            image_z,
            text_z,
            temperature=TEMPERATURE
        )

        # Backward and update
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader):
    model.eval()

    total_loss = 0.0

    for image_x, text_x in loader:

        image_x = image_x.to(device)
        text_x = text_x.to(device)

        image_z, text_z = model(image_x, text_x)

        loss = clip_loss(
            image_z,
            text_z,
            temperature=TEMPERATURE
        )

        total_loss += loss.item()

    return total_loss / len(loader)


# %% Part 5: Full training loop

history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": []
}

best_val_loss = float("inf")
best_epoch = 0
no_improvement = 0

CHECKPOINT_PATH = (
    RESULT_DIR / "best_projection_model.pth"
)
save_config(cfg, RESULT_DIR / "contrastive_config.yaml")

for epoch in range(1, EPOCHS + 1):

    train_loss = train_one_epoch(
        model, train_loader, optimizer
    )

    val_loss = validate(
        model, val_loader
    )

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    # Save best checkpoint
    if val_loss < best_val_loss - training_cfg["min_delta"]:

        best_val_loss = val_loss
        best_epoch = epoch
        no_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "projection_dim": PROJECTION_DIM,
                "temperature": TEMPERATURE,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "seed": SEED,
                "config": cfg,
            },
            CHECKPOINT_PATH
        )

        print("  -> Best checkpoint saved")

    else:
        no_improvement += 1

    # Early stopping
    if no_improvement >= PATIENCE:
        print("Early stopping triggered.")
        break

# Save training history
history_df = pd.DataFrame(history)

history_df.to_csv(
    RESULT_DIR / "training_history.csv",
    index=False
)

print("\nTraining completed.")
print("Best epoch:", best_epoch)
print("Best validation loss:", round(best_val_loss, 4))
print("Checkpoint:", CHECKPOINT_PATH)



# %% Part 6: Plot training curves

history_df = pd.read_csv(
    RESULT_DIR / "training_history.csv"
)

plt.figure(figsize=(8, 5))

plt.plot(
    history_df["epoch"],
    history_df["train_loss"],
    marker="o",
    color="#2166AC",
    linewidth=2,
    label="Train Loss"
)

plt.plot(
    history_df["epoch"],
    history_df["val_loss"],
    marker="s",
    color="#008C79",
    linewidth=2,
    label="Validation Loss"
)

plt.axvline(
    best_epoch,
    color="#777777",
    linestyle="--",
    alpha=0.7,
    label=f"Best Epoch: {best_epoch}"
)

plt.xlabel("Epoch")
plt.ylabel("InfoNCE Loss")
plt.title("Medical CLIP Training Curve")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "training_curve.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("Training curve saved.")

# %%

checkpoint_path = CHECKPOINT_PATH

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

# 1. 查看保存的信息
print("Checkpoint keys:")
print(checkpoint.keys())

# 2. 查看训练轮数与验证损失
print("Epoch:", checkpoint["epoch"])
print("Best Val Loss:", checkpoint["best_val_loss"])

# 3. 查看模型参数
weights = checkpoint["model_state_dict"]

for name, tensor in weights.items():
    print(name, tensor.shape)

