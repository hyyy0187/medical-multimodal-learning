
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoModel


class MedicalCLIP(nn.Module):

    def __init__(
        self, projection_dim=256,
        text_model_name="emilyalsentzer/Bio_ClinicalBERT",
    ):
        super().__init__()

        # 1. Image Encoder: ResNet-18
        self.image_encoder = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )
        self.image_encoder.fc = nn.Identity()

        # 2. Text Encoder: BioClinicalBERT
        self.text_encoder = AutoModel.from_pretrained(
            text_model_name
        )

        # 3. Freeze pretrained encoders
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # 4. Trainable projection heads
        self.image_projection = nn.Linear(
            512, projection_dim
        )
        self.text_projection = nn.Linear(
            768, projection_dim
        )

        self.image_encoder.eval()
        self.text_encoder.eval()

    def train(self, mode=True):
        super().train(mode)

        # Keep frozen encoders in evaluation mode
        self.image_encoder.eval()
        self.text_encoder.eval()

        return self

    def forward(self, images, input_ids, attention_mask):

        # Extract pretrained features
        with torch.no_grad():
            image_features = self.image_encoder(images)

            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            hidden = text_outputs.last_hidden_state

            mask = attention_mask.unsqueeze(-1)
            mask = mask.to(hidden.dtype)

            text_features = (
                (hidden * mask).sum(dim=1)
                / mask.sum(dim=1).clamp(min=1)
            )

        # Project both modalities to the same configured dimension
        image_embeddings = self.image_projection(
            image_features
        )
        text_embeddings = self.text_projection(
            text_features
        )

        # L2 normalization
        image_embeddings = F.normalize(
            image_embeddings, dim=-1
        )
        text_embeddings = F.normalize(
            text_embeddings, dim=-1
        )

        return image_embeddings, text_embeddings
