from __future__ import annotations

from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN

from data.dataset import TrackingDataset
from data.collation import collate_tracking_batch
from detector.config.detector_config import DetectorConfig
from .config import TrainingConfig
from .progress import EpochProgress, EpochProgressCallback
from .checkpoint import CheckpointManager, CheckpointPayload, CheckpointMetadata


class Trainer:
    """Orchestrates model training, device transfers, optimization steps, and checkpointing."""

    def __init__(
        self,
        model: FasterRCNN,
        detector_config: DetectorConfig,
        training_config: TrainingConfig,
        device: torch.device,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ) -> None:
        self.model = model.to(device)
        self.detector_config = detector_config
        self.training_config = training_config
        self.device = device

        self.optimizer = optimizer or torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=training_config.learning_rate,
            momentum=0.9,
            weight_decay=training_config.weight_decay,
        )

        self.scheduler = scheduler or torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=3, gamma=0.1
        )

        self.start_epoch = 0

    def resume_from_checkpoint(self, checkpoint_path: Path) -> CheckpointMetadata:
        """Restores model weights, optimizer momentum, scheduler state, and epoch counter."""
        payload = CheckpointManager.load(checkpoint_path, self.device)

        self.model.load_state_dict(payload.model_state_dict)
        self.optimizer.load_state_dict(payload.optimizer_state_dict)

        if payload.scheduler_state_dict is not None:
            self.scheduler.load_state_dict(payload.scheduler_state_dict)

        self.start_epoch = payload.metadata.epoch + 1
        return payload.metadata

    def train(
        self,
        dataset: TrackingDataset,
        on_progress: Optional[EpochProgressCallback] = None,
    ) -> list[Path]:
        """Runs the training loop and returns paths to saved checkpoints."""
        self.model.train()

        dataloader = DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            collate_fn=collate_tracking_batch,
        )

        total_batches = len(dataloader)
        saved_checkpoints: list[Path] = []

        for epoch in range(self.start_epoch, self.training_config.epochs):
            epoch_loss = 0.0

            for batch_idx, (images, targets) in enumerate(dataloader):
                # 1. Transfer to target hardware
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # 2. Forward pass in train mode returns loss dictionary
                loss_dict = self.model(images, targets)

                # 3. Aggregate all losses (classifier + box_reg + objectness + rpn_box_reg)
                total_loss: torch.Tensor = torch.stack(list(loss_dict.values())).sum()
                loss_val = total_loss.item()
                epoch_loss += loss_val

                # 4. Backward pass and optimizer step
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # 5. Emit progress callback if registered
                if on_progress is not None:
                    on_progress(
                        EpochProgress(
                            epoch=epoch,
                            total_epochs=self.training_config.epochs,
                            batch_index=batch_idx,
                            total_batches=total_batches,
                            loss=loss_val,
                            individual_losses={k: v.item() for k, v in loss_dict.items()},
                        )
                    )

            # Step learning rate scheduler at epoch boundary
            self.scheduler.step()

            avg_epoch_loss = epoch_loss / max(1, total_batches)

            # Checkpoint persistence
            if (epoch + 1) % self.training_config.checkpoint_interval == 0:
                payload = CheckpointPayload(
                    metadata=CheckpointMetadata(
                        epoch=epoch,
                        total_epochs=self.training_config.epochs,
                        loss=avg_epoch_loss,
                        architecture=self.detector_config.architecture.value,
                        num_classes=self.detector_config.num_classes,
                    ),
                    model_state_dict=self.model.state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    scheduler_state_dict=self.scheduler.state_dict(),
                )

                ckpt_path = self.training_config.save_dir / f"checkpoint_epoch_{epoch + 1}.pth"
                saved_path = CheckpointManager.save(payload, ckpt_path)
                saved_checkpoints.append(saved_path)

        return saved_checkpoints
