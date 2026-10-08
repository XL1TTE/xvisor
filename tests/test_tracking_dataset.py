from PIL import Image
import torch
from torch.utils.data import DataLoader

from data.media.types import FrameProvider
from data.annotations.types import FrameAnnotation, ObjectAnnotation, BoundingBox
from data.dataset import TrackingDataset
from data.collation import collate_tracking_batch


def _create_mock_frame_provider(count: int = 3) -> FrameProvider:
    return FrameProvider(
        total_frames=count,
        get_frame=lambda idx: Image.new("RGB", (64, 64), color="blue"),
    )


def test_tracking_dataset_retrieves_image_and_targets() -> None:
    provider = _create_mock_frame_provider(count=2)
    annotations = {
        0: FrameAnnotation(
            frame_index=0,
            objects=(
                ObjectAnnotation(
                    box=BoundingBox(x1=5.0, y1=10.0, x2=20.0, y2=40.0),
                    category_id=1,
                ),
            ),
        ),
        # Frame 1 has no annotations (empty)
        1: FrameAnnotation(frame_index=1, objects=()),
    }

    dataset = TrackingDataset(frame_provider=provider, annotations=annotations)
    assert len(dataset) == 2

    # Verify frame 0
    img0, target0 = dataset[0]
    assert isinstance(img0, torch.Tensor)
    assert img0.shape == (3, 64, 64)
    assert target0["boxes"].shape == (1, 4)
    assert target0["labels"].shape == (1,)
    assert target0["labels"][0].item() == 1
    assert target0["boxes"][0].tolist() == [5.0, 10.0, 20.0, 40.0]

    # Verify frame 1 (empty)
    img1, target1 = dataset[1]
    assert target1["boxes"].shape == (0, 4)
    assert target1["labels"].shape == (0,)


def test_collate_tracking_batch_packages_tuples() -> None:
    provider = _create_mock_frame_provider(count=2)
    annotations = {
        0: FrameAnnotation(
            frame_index=0,
            objects=(
                ObjectAnnotation(
                    box=BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0),
                    category_id=1,
                ),
            ),
        ),
        1: FrameAnnotation(
            frame_index=1,
            objects=(
                ObjectAnnotation(
                    box=BoundingBox(x1=1.0, y1=1.0, x2=5.0, y2=5.0),
                    category_id=1,
                ),
                ObjectAnnotation(
                    box=BoundingBox(x1=10.0, y1=10.0, x2=20.0, y2=20.0),
                    category_id=1,
                ),
            ),
        ),
    }

    dataset = TrackingDataset(frame_provider=provider, annotations=annotations)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_tracking_batch,
    )

    batch_images, batch_targets = next(iter(loader))

    assert isinstance(batch_images, tuple)
    assert len(batch_images) == 2
    assert isinstance(batch_targets, tuple)
    assert len(batch_targets) == 2

    # Frame 0 has 1 box, Frame 1 has 2 boxes
    assert batch_targets[0]["boxes"].shape == (1, 4)
    assert batch_targets[1]["boxes"].shape == (2, 4)
