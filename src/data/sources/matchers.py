from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from typing import Callable, Optional

from .types import InputSource, LocalFile, LocalDirectory, YoutubeUrl, GoogleDriveUri

SourceMatcher = Callable[[str], Optional[InputSource]]


def match_youtube(raw: str) -> YoutubeUrl | None:
    parsed = urlparse(raw.strip())
    domain = parsed.netloc.lower()
    if any(yt in domain for yt in ("youtube.com", "youtu.be", "m.youtube.com")):
        return YoutubeUrl(url=raw.strip())
    return None


def match_google_drive(raw: str) -> GoogleDriveUri | None:
    parsed = urlparse(raw.strip())
    domain = parsed.netloc.lower()
    if "drive.google.com" in domain:
        return GoogleDriveUri(uri=raw.strip())
    return None


def match_local_directory(raw: str) -> LocalDirectory | None:
    path = Path(raw)
    if path.is_dir():
        return LocalDirectory(path=path.resolve())
    return None


def match_local_file(raw: str) -> LocalFile | None:
    path = Path(raw)
    if path.is_file():
        return LocalFile(path=path.resolve())
    return None
