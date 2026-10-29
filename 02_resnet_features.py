
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from experiment import load_config, seed_everything, get_device, project_path

cfg = load_config()
seed_everything(cfg["seed"], deterministic=cfg["runtime"]["deterministic"])

# 1. 选择计算设备
device = get_device(cfg)
print("Device:", device)

# 2. 加载预训练 ResNet-18
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# 3. 去掉最后的分类层，输出512维图像特征
model.fc = nn.Identity()

# 4. 移动到设备，并切换到推理模式
model = model.to(device)
model.eval()

# 5. 模拟两张图像（B, C, H, W）
images = torch.randn(2, 3, 224, 224).to(device)

print("Input shape:", images.shape)

# 6. 提取特征
with torch.no_grad():
    features = model(images)

print("Feature shape:", features.shape)
print("First image feature:", features[0, :10])



from PIL import Image

# 1. 读取真实图像
image = Image.open(project_path(cfg, "demo_image")).convert("RGB")
print("Original image size:", image.size)

# 2. 使用预训练模型对应的图像预处理
preprocess = weights.transforms()
image_tensor = preprocess(image)

print("After preprocessing:", image_tensor.shape)

# 3. 增加 batch 维度
image_tensor = image_tensor.unsqueeze(0).to(device)

print("Model input:", image_tensor.shape)

# 4. 提取图像特征
with torch.no_grad():
    image_features = model(image_tensor)

print("Extracted features:", image_features.shape)
