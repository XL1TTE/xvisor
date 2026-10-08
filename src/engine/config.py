from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 2
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    checkpoint_interval: int = 1
    save_dir: Path = Path("checkpoints")
