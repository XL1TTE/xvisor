from __future__ import annotations

from typing import Callable, Optional
from PIL import Image
import torch
import torchvision.transforms.functional as F


TransformFn = Callable[[Image.Image], torch.Tensor]


def default_transform(image: Image.Image) -> torch.Tensor:
    """Converts a PIL Image to a PyTorch float32 tensor scaled to [0.0, 1.0]."""
    return F.to_tensor(image)
