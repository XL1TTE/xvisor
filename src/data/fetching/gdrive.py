from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from data.sources import GoogleDriveUri, LocalFile
from .progress import ProgressCallback


@dataclass(frozen=True)
class GDriveFetchSuccess:
    resolved: LocalFile


@dataclass(frozen=True)
class GDriveAuthRequired:
    auth_url: str
    message: str


@dataclass(frozen=True)
class GDriveFetchError:
    error_message: str


GDriveFetchResult = Union[
    GDriveFetchSuccess,
    GDriveAuthRequired,
    GDriveFetchError,
]


class GDriveFetcher:
    def __init__(self, cache_dir: Path = Path(".cache/gdrive")) -> None:
        self.cache_dir = cache_dir

    def fetch(
        self,
        source: GoogleDriveUri,
        token: Optional[str] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> GDriveFetchResult:
        # Placeholder for full Google Drive API / OAuth implementation
        return GDriveFetchError(
            f"Google Drive downloading is not yet configured for URI: '{source.uri}'"
        )
