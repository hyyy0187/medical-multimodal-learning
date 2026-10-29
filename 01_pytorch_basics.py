
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from experiment import load_config, seed_everything, get_device, dataloader_options

# 1. Reproducibility
cfg = load_config()
settings = cfg["basics"]
seed_everything(cfg["seed"], deterministic=cfg["runtime"]["deterministic"])

device = get_device(cfg)
print("Device:", device)

# 2. Generate data
X = torch.randn(settings["samples"], settings["input_dim"])
y = (X[:, :settings["informative_features"]].sum(dim=1) > 0).float().unsqueeze(1)

# 3. Split before training
X_train, X_val, y_train, y_val = train_test_split(
    X.numpy(),
    y.numpy(),
    test_size=settings["validation_fraction"],
    random_state=cfg["seed"],
    stratify=y.numpy().ravel()
)

train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    ),
    batch_size=settings["batch_size"],
    shuffle=True,
    **dataloader_options(cfg)
)

X_val = torch.tensor(X_val, dtype=torch.float32).to(device)

# 4. Build model
model = nn.Sequential(
    nn.Linear(settings["input_dim"], settings["hidden_dim"]),
    nn.ReLU(),
    nn.Linear(settings["hidden_dim"], 1)
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])

# 5. Train and evaluate
for epoch in range(settings["epochs"]):
    model.train()
    train_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_X)
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(batch_X)

    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()

    with torch.no_grad():
        val_logits = model(X_val)
        val_probs = torch.sigmoid(val_logits).cpu().numpy().ravel()

        val_loss = criterion(
            val_logits,
            torch.tensor(y_val, dtype=torch.float32).to(device)
        ).item()

    val_preds = (val_probs >= 0.5).astype(int)

    acc = accuracy_score(y_val.ravel(), val_preds)
    auc = roc_auc_score(y_val.ravel(), val_probs)

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Acc: {acc:.4f} | "
        f"AUROC: {auc:.4f}"
    )
