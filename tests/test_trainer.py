from pathlib import Path
from PIL import Image
import torch

from detector import Detector, DetectorConfig, DetectorArchitecture
from data.media.types import FrameProvider
from data.annotations.types import FrameAnnotation, ObjectAnnotation, BoundingBox
from data.dataset import TrackingDataset
from engine import (
    Trainer,
    TrainingConfig,
    CheckpointManager,
    CheckpointPayload,
    CheckpointMetadata,
)


def _create_mock_dataset() -> TrackingDataset:
    provider = FrameProvider(
        total_frames=2,
        get_frame=lambda idx: Image.new("RGB", (64, 64), color="green"),
    )
    annotations = {
        0: FrameAnnotation(
            frame_index=0,
            objects=(
                ObjectAnnotation(
                    box=BoundingBox(x1=5.0, y1=5.0, x2=25.0, y2=25.0),
                    category_id=1,
                ),
            ),
        ),
        1: FrameAnnotation(
            frame_index=1,
            objects=(
                ObjectAnnotation(
                    box=BoundingBox(x1=10.0, y1=10.0, x2=30.0, y2=30.0),
                    category_id=1,
                ),
            ),
        ),
    }
    return TrackingDataset(frame_provider=provider, annotations=annotations)


def test_checkpoint_save_and_load(tmp_path: Path) -> None:
    detector_config = DetectorConfig(
        architecture=DetectorArchitecture.MOBILENET_V3,
        num_classes=2,
        pretrain=False,
    )
    model = Detector.create(detector_config)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    payload = CheckpointPayload(
        metadata=CheckpointMetadata(
            epoch=2,
            total_epochs=5,
            loss=0.45,
            architecture=detector_config.architecture.value,
            num_classes=detector_config.num_classes,
        ),
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
    )

    ckpt_file = tmp_path / "test_checkpoint.pth"
    saved_path = CheckpointManager.save(payload, ckpt_file)
    assert saved_path.exists()

    loaded = CheckpointManager.load(ckpt_file, device=torch.device("cpu"))
    assert loaded.metadata.epoch == 2
    assert loaded.metadata.total_epochs == 5
    assert loaded.metadata.loss == 0.45
    assert loaded.metadata.architecture == "mobilenet_v3"
    assert loaded.metadata.num_classes == 2
    assert "roi_heads.box_predictor.cls_score.weight" in loaded.model_state_dict


def test_trainer_single_epoch_run(tmp_path: Path) -> None:
    device = torch.device("cpu")
    detector_config = DetectorConfig(
        architecture=DetectorArchitecture.MOBILENET_V3,
        num_classes=2,
        pretrain=False,
    )
    model = Detector.create(detector_config)

    training_config = TrainingConfig(
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        save_dir=tmp_path / "ckpts",
        checkpoint_interval=1,
    )

    trainer = Trainer(
        model=model,
        detector_config=detector_config,
        training_config=training_config,
        device=device,
    )

    dataset = _create_mock_dataset()
    saved = trainer.train(dataset)

    assert len(saved) == 1
    assert saved[0].exists()
    assert saved[0].name == "checkpoint_epoch_1.pth"


def test_trainer_resume_session(tmp_path: Path) -> None:
    device = torch.device("cpu")
    detector_config = DetectorConfig(
        architecture=DetectorArchitecture.MOBILENET_V3,
        num_classes=2,
        pretrain=False,
    )
    model = Detector.create(detector_config)

    training_config = TrainingConfig(
        epochs=2,
        batch_size=2,
        save_dir=tmp_path / "resume_ckpts",
    )

    trainer = Trainer(
        model=model,
        detector_config=detector_config,
        training_config=training_config,
        device=device,
    )

    dataset = _create_mock_dataset()
    # Train 1 epoch
    saved = trainer.train(dataset)
    first_ckpt = saved[0]

    # Resume on a new trainer instance
    new_model = Detector.create(detector_config)
    resumed_trainer = Trainer(
        model=new_model,
        detector_config=detector_config,
        training_config=TrainingConfig(epochs=3, save_dir=tmp_path / "resume_ckpts"),
        device=device,
    )

    resumed_meta = resumed_trainer.resume_from_checkpoint(first_ckpt)
    assert resumed_meta.epoch == 0
    assert resumed_trainer.start_epoch == 1

    # Train remaining epochs
    resumed_saved = resumed_trainer.train(dataset)
    assert len(resumed_saved) == 2  # Epoch 2 and Epoch 3 saved
