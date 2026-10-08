from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box in absolute pixel coordinates [x1, y1, x2, y2]."""
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class ObjectAnnotation:
    box: BoundingBox
    category_id: int = 1
    track_id: Optional[int] = None


@dataclass(frozen=True)
class FrameAnnotation:
    frame_index: int
    objects: tuple[ObjectAnnotation, ...]


# Format Discriminators
@dataclass(frozen=True)
class CocoJsonAnnotation:
    file_path: Path


@dataclass(frozen=True)
class MotTxtAnnotation:
    file_path: Path


AnnotationFormat = Union[CocoJsonAnnotation, MotTxtAnnotation]
