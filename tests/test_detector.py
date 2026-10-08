import pytest
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from detector import Detector, DetectorConfig, DetectorArchitecture


@pytest.mark.parametrize(
    "architecture",
    [
        DetectorArchitecture.MOBILENET_V3,
        DetectorArchitecture.RESNET_50,
    ],
)
def test_detector_replaces_box_predictor_head(architecture: DetectorArchitecture) -> None:
    expected_classes = 2
    config = DetectorConfig(
        architecture=architecture,
        num_classes=expected_classes,
        pretrain=False,
    )

    model = Detector.create(config)

    assert isinstance(model, FasterRCNN)
    assert isinstance(model.roi_heads.box_predictor, FastRCNNPredictor)
    assert model.roi_heads.box_predictor.cls_score.out_features == expected_classes
