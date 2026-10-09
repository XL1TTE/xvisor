from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SessionConfig:
    architecture: str
    num_classes: int = 2
    name: str = "default_session"


@dataclass(frozen=True)
class SessionMetadata:
    session_id: str
    name: str
    architecture: str
    num_classes: int
    created_at: str
    current_epoch: int = 0
    total_epochs: int = 0
    last_loss: Optional[float] = None
    best_loss: Optional[float] = None
    status: str = "created"
