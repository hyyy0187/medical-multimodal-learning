
# %% Part 1: Data preparation

import copy
import numpy as np
import torch
import torch.nn as nn

from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from experiment import (
    PROJECT_ROOT, load_config, seed_everything, get_device,
    project_path, dataloader_options, save_config,
)

# Set random seed
cfg = load_config()
settings = cfg["pneumonia"]
seed_everything(cfg["seed"], deterministic=cfg["runtime"]["deterministic"])

device = get_device(cfg)
print("Device:", device)

# Preprocessing
weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()

# Training data
full_dataset = ImageFolder(
    project_path(cfg, "chest_xray_dir") / "train",
    transform=preprocess
)

# Stratified train/validation split
indices = np.arange(len(full_dataset))

train_idx, val_idx = train_test_split(
    indices,
    test_size=settings["validation_fraction"],
    random_state=cfg["seed"],
    stratify=full_dataset.targets
)

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)

# Independent test set
test_dataset = ImageFolder(
    project_path(cfg, "chest_xray_dir") / "test",
    transform=preprocess
)

assert full_dataset.class_to_idx == test_dataset.class_to_idx

train_loader = DataLoader(
    train_dataset,
    batch_size=settings["batch_size"],
    shuffle=True,
    **dataloader_options(cfg)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=settings["batch_size"],
    shuffle=False,
    **dataloader_options(cfg)
)

test_loader = DataLoader(
    test_dataset,
    batch_size=settings["batch_size"],
    shuffle=False,
    **dataloader_options(cfg)
)

print("Classes:", full_dataset.class_to_idx)
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))


# %% Part 2: Build model

model = resnet18(weights=weights)

# Freeze pretrained parameters
for param in model.parameters():
    param.requires_grad = False

# Replace final classification layer
model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimize only the classification layer
optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=settings["learning_rate"]
)

print(model.fc)
print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters()
        if p.requires_grad)
)


# %% Part 3: Evaluation function

def evaluate(model, loader):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    return acc, auc, all_labels, all_preds


# %% Part 4: Training

epochs = settings["epochs"]
best_auc = -1
best_state = None

for epoch in range(epochs):

    # Keep pretrained BatchNorm statistics unchanged
    model.eval()
    model.fc.train()

    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_dataset)
    val_acc, val_auc, _, _ = evaluate(model, val_loader)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {train_loss:.4f} | "
        f"Val Accuracy: {val_acc:.4f} | "
        f"Val AUROC: {val_auc:.4f}"
    )

    if val_auc > best_auc:
        best_auc = val_auc
        best_state = copy.deepcopy(model.state_dict())

# Restore best validation model
model.load_state_dict(best_state)
torch.save(best_state, PROJECT_ROOT / "best_pneumonia_resnet18.pth")
save_config(cfg, project_path(cfg, "results_dir") / "pneumonia_config.yaml")

print("Best validation AUROC:", best_auc)


# %% Part 5: Final testing

test_acc, test_auc, y_true, y_pred = evaluate(
    model,
    test_loader
)

print("Test Accuracy:", round(test_acc, 4))
print("Test AUROC:", round(test_auc, 4))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)


# %% Part 6: Detailed evaluation

from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    confusion_matrix
)

# Use results from the independent test set
test_acc, test_auc, y_true, y_pred = evaluate(
    model, test_loader
)

print("Classification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["NORMAL", "PNEUMONIA"],
        digits=4
    )
)

print(
    "Balanced Accuracy:",
    round(balanced_accuracy_score(y_true, y_pred), 4)
)

cm = confusion_matrix(y_true, y_pred)

tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print("Sensitivity:", round(sensitivity, 4))
print("Specificity:", round(specificity, 4))

# %%
