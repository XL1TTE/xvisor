from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Union
from PIL import Image


@dataclass(frozen=True)
class ImageDirectoryMedia:
    directory_path: Path
    image_paths: tuple[Path, ...]


@dataclass(frozen=True)
class VideoFileMedia:
    file_path: Path


@dataclass(frozen=True)
class ArchiveMedia:
    archive_path: Path


MediaType = Union[ImageDirectoryMedia, VideoFileMedia, ArchiveMedia]


@dataclass(frozen=True)
class FrameProvider:
    total_frames: int
    get_frame: Callable[[int], Image.Image]
