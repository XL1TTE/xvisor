from pathlib import Path
from PIL import Image
import torch

from detector import Detector, DetectorConfig, DetectorArchitecture
from data.media.types import FrameProvider
from data.annotations.types import FrameAnnotation, ObjectAnnotation, BoundingBox
from data.dataset import TrackingDataset
from engine.evaluation import (
    compute_iou,
    calculate_ap_for_iou,
    DetectorEvaluator,
    EvaluationConfig,
)


def test_compute_iou_identical_boxes() -> None:
    boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
    boxes2 = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)

    iou = compute_iou(boxes1, boxes2)
    assert iou.shape == (1, 1)
    assert abs(iou[0, 0].item() - 1.0) < 1e-5


def test_compute_iou_disjoint_boxes() -> None:
    boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
    boxes2 = torch.tensor([[20.0, 20.0, 30.0, 30.0]], dtype=torch.float32)

    iou = compute_iou(boxes1, boxes2)
    assert iou.shape == (1, 1)
    assert iou[0, 0].item() == 0.0


def test_calculate_ap_perfect_predictions() -> None:
    # Target: 1 box at [10, 10, 50, 50]
    targets = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]])}]
    # Prediction: Exactly matching box with high confidence
    predictions = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "scores": torch.tensor([0.95]),
        }
    ]

    ap, prec, rec = calculate_ap_for_iou(predictions, targets, iou_threshold=0.50)
    assert ap == 1.0
    assert prec == 1.0
    assert rec == 1.0


def test_detector_evaluator_runs_on_dataset() -> None:
    device = torch.device("cpu")
    model = Detector.create(
        DetectorConfig(
            architecture=DetectorArchitecture.MOBILENET_V3,
            num_classes=2,
            pretrain=False,
        )
    )

    provider = FrameProvider(
        total_frames=1,
        get_frame=lambda idx: Image.new("RGB", (64, 64), color="blue"),
    )
    annotations = {
        0: FrameAnnotation(
            frame_index=0,
            objects=(
                ObjectAnnotation(
                    box=BoundingBox(x1=5.0, y1=5.0, x2=20.0, y2=20.0),
                    category_id=1,
                ),
            ),
        )
    }
    dataset = TrackingDataset(frame_provider=provider, annotations=annotations)

    evaluator = DetectorEvaluator(
        model=model,
        device=device,
        config=EvaluationConfig(batch_size=1, confidence_threshold=0.0),
    )

    metrics = evaluator.evaluate(dataset)

    assert isinstance(metrics.ap50, float)
    assert isinstance(metrics.map50_95, float)
    assert metrics.total_ground_truth == 1
    assert metrics.total_predictions >= 0
