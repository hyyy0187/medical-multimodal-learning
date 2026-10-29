
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from experiment import load_config, seed_everything, get_device, project_path

cfg = load_config()
seed_everything(cfg["seed"], deterministic=cfg["runtime"]["deterministic"])

# 1. Select the compute device
device = get_device(cfg)
print("Device:", device)

# 2. Load the pretrained ResNet-18 model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# 3. Remove the final classification layer to output 512-dimensional image features
model.fc = nn.Identity()

# 4. Move the model to the device and switch to evaluation mode
model = model.to(device)
model.eval()

# 5. Generate two synthetic images (B, C, H, W)
images = torch.randn(2, 3, 224, 224).to(device)

print("Input shape:", images.shape)

# 6. Extract features
with torch.no_grad():
    features = model(images)

print("Feature shape:", features.shape)
print("First image feature:", features[0, :10])



from PIL import Image

# 1. Load a real image
image = Image.open(project_path(cfg, "demo_image")).convert("RGB")
print("Original image size:", image.size)

# 2. Apply the image preprocessing associated with the pretrained weights
preprocess = weights.transforms()
image_tensor = preprocess(image)

print("After preprocessing:", image_tensor.shape)

# 3. Add the batch dimension
image_tensor = image_tensor.unsqueeze(0).to(device)

print("Model input:", image_tensor.shape)

# 4. Extract image features
with torch.no_grad():
    image_features = model(image_tensor)

print("Extracted features:", image_features.shape)
