from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from data.sources import YoutubeUrl, LocalFile
from .progress import ProgressCallback


@dataclass(frozen=True)
class YoutubeFetchSuccess:
    resolved: LocalFile


@dataclass(frozen=True)
class YoutubeFetchError:
    error_message: str


YoutubeFetchResult = Union[YoutubeFetchSuccess, YoutubeFetchError]


class YoutubeFetcher:
    def __init__(self, cache_dir: Path = Path(".cache/youtube")) -> None:
        self.cache_dir = cache_dir

    def fetch(
        self,
        source: YoutubeUrl,
        on_progress: Optional[ProgressCallback] = None,
    ) -> YoutubeFetchResult:
        # Placeholder for full yt-dlp implementation
        return YoutubeFetchError(
            f"YouTube downloading is not yet configured for URL: '{source.url}'"
        )
