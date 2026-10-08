from __future__ import annotations

from pathlib import Path
from data.sources import LocalFile, LocalDirectory
from .types import MediaType, ImageDirectoryMedia, VideoFileMedia, ArchiveMedia

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".tar.gz", ".7z"}


def classify_media(source: LocalFile | LocalDirectory) -> MediaType:
    match source:
        case LocalDirectory(path):
            images = tuple(
                sorted(
                    p.resolve()
                    for p in path.iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS
                )
            )
            if not images:
                raise ValueError(f"Directory contains no supported image frames: '{path}'")
            return ImageDirectoryMedia(directory_path=path, image_paths=images)

        case LocalFile(path):
            ext = path.suffix.lower()
            if ext in VIDEO_EXTENSIONS:
                return VideoFileMedia(file_path=path)
            elif ext in ARCHIVE_EXTENSIONS:
                return ArchiveMedia(archive_path=path)
            else:
                raise ValueError(f"Unsupported media file format: '{ext}' for file '{path}'")
