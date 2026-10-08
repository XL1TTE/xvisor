from __future__ import annotations

from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset

from .media.types import FrameProvider
from .annotations.types import FrameAnnotation
from .transforms import TransformFn, default_transform


class TrackingDataset(Dataset):
    """Dataset providing images and bounding box annotations for object detection."""

    def __init__(
        self,
        frame_provider: FrameProvider,
        annotations: dict[int, FrameAnnotation],
        transform: Optional[TransformFn] = None,
    ) -> None:
        self.frame_provider = frame_provider
        self.annotations = annotations
        self.transform = transform or default_transform

    def __len__(self) -> int:
        return self.frame_provider.total_frames

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of bounds for dataset of length {len(self)}")

        pil_image = self.frame_provider.get_frame(index)
        image_tensor = self.transform(pil_image)

        frame_ann = self.annotations.get(index)

        if frame_ann is not None and frame_ann.objects:
            boxes = [
                [obj.box.x1, obj.box.y1, obj.box.x2, obj.box.y2]
                for obj in frame_ann.objects
            ]
            labels = [obj.category_id for obj in frame_ann.objects]

            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
        }

        return image_tensor, target
