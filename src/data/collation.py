from __future__ import annotations

from typing import Any, Sequence, Tuple
import torch


def collate_tracking_batch(
    batch: Sequence[Tuple[torch.Tensor, dict[str, torch.Tensor]]]
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[dict[str, torch.Tensor], ...]]:
    """Custom collate function for object detection datasets.

    Faster R-CNN requires images and targets to be passed as sequences (tuples or lists),
    because each image can contain a variable number of bounding boxes.
    """
    images, targets = zip(*batch)
    return tuple(images), tuple(targets)
