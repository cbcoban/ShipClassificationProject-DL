# src/model.py
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    EfficientNet-B0 backbone + sınıflandırma başlığı.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)

    # EfficientNet classifier: Sequential(Dropout, Linear)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def predict_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model(x)
