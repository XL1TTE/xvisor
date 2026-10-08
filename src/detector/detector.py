from typing import cast
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from model_provider import ModelProvider
from .config.detector_config import DetectorConfig


class Detector:
    @staticmethod
    def create(config: DetectorConfig) -> FasterRCNN:
        model = ModelProvider.resolve_model(config)

        predictor = cast(FastRCNNPredictor, model.roi_heads.box_predictor)
        in_features = predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config.num_classes)

        return model
