from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class ProgressUpdate:
    downloaded_bytes: int
    total_bytes: Optional[int] = None
    speed_bytes_per_sec: Optional[float] = None


ProgressCallback = Callable[[ProgressUpdate], None]
