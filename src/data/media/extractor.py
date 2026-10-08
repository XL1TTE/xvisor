from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path
from PIL import Image

from .types import (
    FrameProvider,
    MediaType,
    ImageDirectoryMedia,
    VideoFileMedia,
    ArchiveMedia,
)
from .classifier import IMAGE_EXTENSIONS


def extract_from_directory(media: ImageDirectoryMedia) -> FrameProvider:
    paths = media.image_paths
    return FrameProvider(
        total_frames=len(paths),
        get_frame=lambda idx: Image.open(paths[idx]).convert("RGB"),
    )


def extract_from_video(media: VideoFileMedia) -> FrameProvider:
    import cv2

    cap = cv2.VideoCapture(str(media.file_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: '{media.file_path}'")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame(idx: int) -> Image.Image:
        if idx < 0 or idx >= total:
            raise IndexError(f"Frame index {idx} out of range [0, {total})")

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success or frame is None:
            raise IndexError(f"Failed to read frame {idx} from video: '{media.file_path}'")

        # OpenCV decodes as BGR; convert to standard RGB PIL Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    return FrameProvider(total_frames=total, get_frame=read_frame)


def extract_from_archive(media: ArchiveMedia) -> FrameProvider:
    archive = zipfile.ZipFile(media.archive_path, "r")
    names = tuple(
        sorted(
            n
            for n in archive.namelist()
            if Path(n).suffix.lower() in IMAGE_EXTENSIONS and not n.startswith("__MACOSX/")
        )
    )
    if not names:
        raise ValueError(f"Archive contains no supported image frames: '{media.archive_path}'")

    def read_frame(idx: int) -> Image.Image:
        if idx < 0 or idx >= len(names):
            raise IndexError(f"Frame index {idx} out of range [0, {len(names)})")

        with archive.open(names[idx]) as stream:
            return Image.open(BytesIO(stream.read())).convert("RGB")

    return FrameProvider(total_frames=len(names), get_frame=read_frame)


def resolve_frame_provider(media: MediaType) -> FrameProvider:
    match media:
        case ImageDirectoryMedia() as m:
            return extract_from_directory(m)
        case VideoFileMedia() as m:
            return extract_from_video(m)
        case ArchiveMedia() as m:
            return extract_from_archive(m)
