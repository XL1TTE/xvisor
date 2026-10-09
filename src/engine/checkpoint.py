from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional
import torch

from detector.config.detector_architecture import DetectorArchitecture
from detector.config.detector_config import DetectorConfig


@dataclass(frozen=True)
class CheckpointMetadata:
    epoch: int
    total_epochs: int
    loss: float
    architecture: str
    num_classes: int


@dataclass(frozen=True)
class CheckpointPayload:
    metadata: CheckpointMetadata
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: Optional[dict[str, Any]] = None


class CheckpointManager:
    """Manages serialization and restoration of complete training sessions."""

    @staticmethod
    def save(payload: CheckpointPayload, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        raw_dict = {
            "metadata": asdict(payload.metadata),
            "model_state_dict": payload.model_state_dict,
            "optimizer_state_dict": payload.optimizer_state_dict,
            "scheduler_state_dict": payload.scheduler_state_dict,
        }

        torch.save(raw_dict, output_path)
        return output_path

    @staticmethod
    def load(checkpoint_path: Path, device: torch.device) -> CheckpointPayload:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file does not exist: '{checkpoint_path}'")

        raw_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

        meta_dict = raw_dict["metadata"]
        metadata = CheckpointMetadata(
            epoch=meta_dict["epoch"],
            total_epochs=meta_dict["total_epochs"],
            loss=meta_dict["loss"],
            architecture=meta_dict["architecture"],
            num_classes=meta_dict["num_classes"],
        )

        return CheckpointPayload(
            metadata=metadata,
            model_state_dict=raw_dict["model_state_dict"],
            optimizer_state_dict=raw_dict["optimizer_state_dict"],
            scheduler_state_dict=raw_dict.get("scheduler_state_dict"),
        )

    @staticmethod
    def export_weights_only(checkpoint_path: Path, output_path: Path) -> Path:
        """Strips optimizer and scheduler states, producing a lightweight inference model."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file does not exist: '{checkpoint_path}'")

        raw_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(raw_dict["model_state_dict"], output_path)
        return output_path
