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


# Tagged union for local input sources
InputSource = Union[LocalFile, LocalDirectory]
