from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from data.sources import LocalFile, LocalDirectory


@dataclass(frozen=True)
class LocalFetchSuccess:
    resolved: LocalFile | LocalDirectory


@dataclass(frozen=True)
class LocalFetchError:
    error_message: str


LocalFetchResult = Union[LocalFetchSuccess, LocalFetchError]


class LocalFetcher:
    @staticmethod
    def fetch(source: LocalFile | LocalDirectory) -> LocalFetchResult:
        path = source.path
        if not path.exists():
            return LocalFetchError(f"Path does not exist on disk: '{path}'")
        return LocalFetchSuccess(resolved=source)
