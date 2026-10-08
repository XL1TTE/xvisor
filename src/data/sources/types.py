from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class LocalFile:
    path: Path


@dataclass(frozen=True)
class LocalDirectory:
    path: Path


@dataclass(frozen=True)
class YoutubeUrl:
    url: str


@dataclass(frozen=True)
class GoogleDriveUri:
    uri: str


# Tagged Union of all recognized input sources
InputSource = Union[LocalFile, LocalDirectory, YoutubeUrl, GoogleDriveUri]
