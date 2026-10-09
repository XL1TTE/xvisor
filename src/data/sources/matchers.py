from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .types import InputSource, LocalFile, LocalDirectory

SourceMatcher = Callable[[str], Optional[InputSource]]


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
