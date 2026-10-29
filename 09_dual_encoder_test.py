# %% Part 1: Forward and backward test
import torch
from transformers import AutoTokenizer

from models import MedicalCLIP
from losses import clip_loss
from experiment import load_config, seed_everything, get_device, project_path

# 1. Set random seed
cfg = load_config()
seed_everything(cfg["seed"], deterministic=cfg["runtime"]["deterministic"])

device = get_device(cfg)

print("Device:", device)

# 2. Load dual encoder
model = MedicalCLIP(
    projection_dim=cfg["models"]["projection_dim"],
    text_model_name=cfg["models"]["text_name"]
).to(device)

model.train()

# 3. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    cfg["models"]["text_name"]
)

# 4. Create synthetic images
images = torch.randn(
    2, 3, 224, 224,
    device=device
)

# 5. Create example texts
texts = [
    "The lungs are clear.",
    "Right lower lobe opacity is present."
]

encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=cfg["text"]["demo_max_length"],
    return_tensors="pt"
).to(device)

# 6. Forward propagation
image_z, text_z = model(
    images,
    encoded["input_ids"],
    encoded["attention_mask"]
)

print("Image embeddings:", image_z.shape)
print("Text embeddings:", text_z.shape)

# 7. Calculate InfoNCE Loss
loss = clip_loss(
    image_z,
    text_z,
    temperature=cfg["contrastive"]["temperature"],
    uids=["demo_1", "demo_2"]
)

print("InfoNCE Loss:", loss.item())

# 8. Backward propagation
model.zero_grad(set_to_none=True)

loss.backward()

# Check gradients of both projection layers
image_grad = model.image_projection.weight.grad
text_grad = model.text_projection.weight.grad

assert image_grad is not None
assert text_grad is not None
assert torch.isfinite(loss).item()
assert torch.isfinite(image_grad).all().item()
assert torch.isfinite(text_grad).all().item()

print(
    "Image projection gradient:",
    image_grad.norm().item()
)

print(
    "Text projection gradient:",
    text_grad.norm().item()
)

print("Forward and backward test: PASSED")

# %% Part 2: Train on one real image-text batch

import pandas as pd
from PIL import Image
from torchvision.models import ResNet18_Weights

# 1. Data paths
DATA_DIR = project_path(cfg, "iu_xray_dir")

# 2. Read training pairs
train_df = pd.read_csv(
    DATA_DIR / "train_pairs.csv",
    dtype={"uid": str}
)

print("\nOriginal training pairs:", len(train_df))

# 3. Select 4 unique reports
# Prevent duplicate UIDs in one contrastive batch
samples = (
    train_df
    .drop_duplicates(subset="uid")
    .head(4)
)

assert len(samples) == 4

print("Selected UIDs:", samples["uid"].tolist())

# 4. Image preprocessing
preprocess = ResNet18_Weights.DEFAULT.transforms()

image_list = []

for _, row in samples.iterrows():

    image_path = DATA_DIR / row["image_path"]

    with Image.open(image_path) as img:
        image_tensor = preprocess(
            img.convert("RGB")
        )

    image_list.append(image_tensor)

images = torch.stack(image_list).to(device)

# 5. Text tokenization
texts = samples["report"].tolist()

encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=cfg["text"]["max_length"],
    return_tensors="pt"
).to(device)

print("Real images:", images.shape)
print("Real input IDs:", encoded["input_ids"].shape)

# 6. Optimizer: projection layers only
optimizer = torch.optim.AdamW(
    (p for p in model.parameters()
     if p.requires_grad),
    lr=cfg["contrastive"]["learning_rate"]
)

# 7. Training step
model.train()
optimizer.zero_grad(set_to_none=True)

# Save weights before updating
old_weights = (
    model.image_projection.weight
    .detach().clone()
)

# Forward
image_z, text_z = model(
    images,
    encoded["input_ids"],
    encoded["attention_mask"]
)

# Loss
loss = clip_loss(
    image_z,
    text_z,
    temperature=cfg["contrastive"]["temperature"],
    uids=samples["uid"].tolist()
)

print("Training Loss:", loss.item())

# Backward
loss.backward()

# Update trainable parameters
optimizer.step()

# 8. Verify weight update
weight_change = (
    model.image_projection.weight.detach()
    - old_weights
).abs().sum().item()

print("Weight change:", weight_change)

assert torch.isfinite(loss).item()
assert weight_change > 0

print("One real training step: PASSED")
