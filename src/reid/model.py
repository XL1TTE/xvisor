from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class ReIDModel(nn.Module):
    """Deep visual appearance feature extractor for cropped person patches.

    Inputs:
        Tensor of shape [Batch, 3, 128, 64]
    Outputs:
        L2-normalized feature embeddings of shape [Batch, 576]
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base_model = mobilenet_v3_small(weights=weights)

        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)

        # L2-normalization so dot product in downstream apps equals cosine similarity
        norm = torch.norm(feat, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        return feat / norm
