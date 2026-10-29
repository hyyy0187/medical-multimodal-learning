
# %% Part 1: Data preparation

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader

from losses import clip_loss
from experiment import (
    load_config, project_path, seed_everything, get_device,
    dataloader_options, save_config,
)

cfg = load_config()
finetune_cfg = cfg["finetune"]
DATA_DIR = project_path(cfg, "iu_xray_dir")
RESULT_DIR = project_path(cfg, "results_dir")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED = cfg["seed"]
seed_everything(SEED, deterministic=cfg["runtime"]["deterministic"])

device = get_device(cfg)

# True = 小规模测试；False = 完整训练
PILOT = finetune_cfg["pilot"]

BATCH_SIZE = finetune_cfg["batch_size"]
EPOCHS = finetune_cfg["pilot_epochs"] if PILOT else finetune_cfg["epochs"]

# Load best checkpoint
checkpoint = torch.load(
    RESULT_DIR / "best_projection_model.pth",
    map_location="cpu",
    weights_only=True
)

print("Loaded best epoch:", checkpoint["epoch"])
# Preserve the architecture and temperature used by the starting checkpoint.
PROJECTION_DIM = checkpoint.get(
    "projection_dim", cfg["models"]["projection_dim"]
)
TEMPERATURE = checkpoint.get(
    "temperature", cfg["contrastive"]["temperature"]
)
# Resolve configuration before recording it in snapshots and new checkpoints.
cfg["models"]["projection_dim"] = PROJECTION_DIM
cfg["contrastive"]["temperature"] = TEMPERATURE
cfg["loaded_checkpoint"] = {
    "path": str(RESULT_DIR / "best_projection_model.pth"),
    "projection_dim": PROJECTION_DIM,
    "projection_dim_source": (
        "checkpoint" if "projection_dim" in checkpoint else "config.yaml"
    ),
    "temperature": TEMPERATURE,
    "temperature_source": (
        "checkpoint" if "temperature" in checkpoint else "config.yaml"
    ),
}
print("Checkpoint projection dimension:", PROJECTION_DIM)
print("Checkpoint temperature:", TEMPERATURE)

# Load BERT features
text_data = np.load(
    DATA_DIR / "iu_bert_report_features.npz"
)

text_features = text_data["features"].astype(np.float32)
text_uids = text_data["report_uids"].astype(str)

preprocess = ResNet18_Weights.DEFAULT.transforms()

# Read train and validation splits
def load_unique_pairs(filename):
    df = pd.read_csv(
        DATA_DIR / filename,
        dtype={"uid": str, "filename": str}
    )

    return (
        df.sort_values(["uid", "filename"])
          .drop_duplicates(subset="uid")
          .reset_index(drop=True)
    )

train_df = load_unique_pairs("train_pairs.csv")
val_df = load_unique_pairs("val_pairs.csv")

assert set(train_df["uid"]).isdisjoint(set(val_df["uid"]))

if PILOT:
    train_df = train_df.sample(
        n=min(finetune_cfg["pilot_samples"], len(train_df)),
        random_state=SEED
    ).reset_index(drop=True)

class PairedDataset(Dataset):

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = DATA_DIR / row["image_path"]

        with Image.open(path) as img:
            image = preprocess(img.convert("RGB"))

        text_row = int(row["feature_row"])

        assert text_uids[text_row] == str(row["uid"])

        text_vector = torch.from_numpy(
            text_features[text_row].copy()
        )

        return image, text_vector, str(row["uid"])


train_loader = DataLoader(
    PairedDataset(train_df),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    **dataloader_options(cfg),
)

val_loader = DataLoader(
    PairedDataset(val_df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    **dataloader_options(cfg),
)

print("Device:", device)
print("Train pairs:", len(train_df))
print("Validation pairs:", len(val_df))


# %% Part 2: Partial fine-tuning model

class PartialFineTuneCLIP(nn.Module):

    def __init__(self, checkpoint):
        super().__init__()

        projection_dim = PROJECTION_DIM

        # Image encoder
        self.image_encoder = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )
        self.image_encoder.fc = nn.Identity()

        # Freeze the entire image encoder
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        # Unfreeze only layer4
        for p in self.image_encoder.layer4.parameters():
            p.requires_grad = True

        # Projection heads
        self.image_projection = nn.Linear(
            512, projection_dim
        )
        self.text_projection = nn.Linear(
            768, projection_dim
        )

        # Restore Day 5 projection weights
        state = checkpoint["model_state_dict"]

        self.image_projection.load_state_dict({
            "weight": state["image_projection.weight"],
            "bias": state["image_projection.bias"]
        })

        self.text_projection.load_state_dict({
            "weight": state["text_projection.weight"],
            "bias": state["text_projection.bias"]
        })

    def train(self, mode=True):
        super().train(mode)

        # Keep pretrained BatchNorm statistics fixed.
        # eval() does not disable gradients for layer4.
        self.image_encoder.eval()

        return self

    def forward(self, images, text_features):

        # Do not use torch.no_grad() here!
        image_features = self.image_encoder(images)

        image_z = self.image_projection(
            image_features
        )

        text_z = self.text_projection(
            text_features
        )

        image_z = F.normalize(image_z, dim=-1)
        text_z = F.normalize(text_z, dim=-1)

        return image_z, text_z


model = PartialFineTuneCLIP(checkpoint).to(device)

# Different learning rates for pretrained
# and newly trained components
optimizer = torch.optim.AdamW(
    [
        {
            "params": model.image_encoder.layer4.parameters(),
            "lr": finetune_cfg["encoder_learning_rate"]
        },
        {
            "params": model.image_projection.parameters(),
            "lr": finetune_cfg["projection_learning_rate"]
        },
        {
            "params": model.text_projection.parameters(),
            "lr": finetune_cfg["projection_learning_rate"]
        }
    ],
    weight_decay=finetune_cfg["weight_decay"]
)

print("Trainable parameters:")
for name, p in model.named_parameters():
    if p.requires_grad:
        print(name, tuple(p.shape))



# %% Part 3: Retrieval metrics and training

def get_retrieval_metrics(similarity):
    n = similarity.shape[0]
    output = {}

    for direction, scores in [
        ("I2T", similarity),
        ("T2I", similarity.T)
    ]:
        order = np.argsort(
            -scores, axis=1, kind="stable"
        )

        correct = np.arange(n)[:, None]
        ranks = np.argmax(order == correct, axis=1) + 1

        output[direction] = {
            "R@1": np.mean(ranks <= 1) * 100,
            "R@5": np.mean(ranks <= 5) * 100,
            "R@10": np.mean(ranks <= 10) * 100,
            "MedR": float(np.median(ranks))
        }

    return output


@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()

    image_vectors = []
    text_vectors = []

    for images, text_x, uids in loader:
        images = images.to(device)
        text_x = text_x.to(device)

        image_z, text_z = model(images, text_x)

        image_vectors.append(image_z.cpu())
        text_vectors.append(text_z.cpu())

    image_z = torch.cat(image_vectors, dim=0)
    text_z = torch.cat(text_vectors, dim=0)

    similarity = (image_z @ text_z.T).numpy()

    return get_retrieval_metrics(similarity)


def mean_r5(metrics):
    return (
        metrics["I2T"]["R@5"] +
        metrics["T2I"]["R@5"]
    ) / 2


# Evaluate checkpoint before fine-tuning
baseline = evaluate_model(model, val_loader)

print("Before fine-tuning:", baseline)

best_score = mean_r5(baseline)
best_epoch = 0

save_name = (
    "pilot_partial_finetune.pth"
    if PILOT else "best_partial_finetune.pth"
)
save_path = RESULT_DIR / save_name
save_config(
    cfg,
    save_path.with_suffix(".yaml"),
)

def save_checkpoint(epoch, score):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "mean_r5": float(score),
            "pilot": PILOT,
            "projection_dim": PROJECTION_DIM,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "config": cfg,
        },
        save_path
    )

# Epoch 0 means the original model
save_checkpoint(0, best_score)

for epoch in range(1, EPOCHS + 1):

    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, text_x, uids in train_loader:

        images = images.to(device)
        text_x = text_x.to(device)

        optimizer.zero_grad(set_to_none=True)

        image_z, text_z = model(images, text_x)

        loss = clip_loss(
            image_z,
            text_z,
            temperature=TEMPERATURE,
            uids=uids
        )

        loss.backward()

        # Verify gradients reach layer4
        grad = model.image_encoder.layer4[
            1
        ].conv2.weight.grad

        assert grad is not None
        assert torch.isfinite(grad).all()

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    train_loss = total_loss / n_batches
    metrics = evaluate_model(model, val_loader)
    score = mean_r5(metrics)

    print(
        f"Epoch {epoch} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Mean Val R@5: {score:.3f}%"
    )

    print("I2T:", metrics["I2T"])
    print("T2I:", metrics["T2I"])

    if score > best_score + 1e-8:
        best_score = score
        best_epoch = epoch
        save_checkpoint(epoch, score)

        print("  -> New best checkpoint saved")

print("Best epoch:", best_epoch)
print("Best mean R@5:", best_score)
print("Saved:", save_path)

# %%
