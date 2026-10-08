from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class EpochProgress:
    epoch: int
    total_epochs: int
    batch_index: int
    total_batches: int
    loss: float
    individual_losses: dict[str, float]


EpochProgressCallback = Callable[[EpochProgress], None]
