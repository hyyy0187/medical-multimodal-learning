
# %% 第一部分：导入库并加载模型

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from pathlib import Path
from experiment import (
    PROJECT_ROOT, load_config, seed_everything, get_device,
    project_path, dataloader_options,
)

cfg = load_config()
seed_everything(cfg["seed"], deterministic=cfg["runtime"]["deterministic"])
device = get_device(cfg)

weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()

model = resnet18(weights=weights)
model.fc = nn.Identity()
model = model.to(device)
model.eval()


# %% 第二部分：创建 Dataset

class ChestXrayDataset(Dataset):

    def __init__(self, image_dir, transform):
        self.image_paths = sorted(
            p for p in Path(image_dir).iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        image = self.transform(image)

        return image, image_path.name


# %% 第三部分：批量提取特征

dataset = ChestXrayDataset(
    image_dir=PROJECT_ROOT,
    transform=preprocess
)

print("Number of images:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    **dataloader_options(cfg)
)

all_features = []
all_names = []

with torch.no_grad():
    for images, names in loader:
        images = images.to(device)
        features = model(images)

        print("Batch input:", images.shape)
        print("Batch features:", features.shape)

        all_features.append(features.cpu())
        all_names.extend(names)

if all_features:
    all_features = torch.cat(all_features, dim=0)
    print("Total feature matrix:", all_features.shape)
    print("Image filenames:", all_names)

# %%

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

dataset = ImageFolder(
    root=project_path(cfg, "chest_xray_dir") / "train",
    transform=preprocess
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    **dataloader_options(cfg)
)

print("Number of images:", len(dataset))
print("Classes:", dataset.class_to_idx)

# 先测试一个 Batch
images, labels = next(iter(loader))
images = images.to(device)

with torch.no_grad():
    features = model(images)

print("Images:", images.shape)
print("Labels:", labels)
print("Features:", features.shape)

# %%
