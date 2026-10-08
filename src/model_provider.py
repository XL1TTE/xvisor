from torchvision.models.detection import (
    FasterRCNN,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn_v2,
)

from detector.config.detector_architecture import DetectorArchitecture
from detector.config.detector_config import DetectorConfig


class ModelProvider:
    @staticmethod
    def resolve_model(config: DetectorConfig) -> FasterRCNN:
        weights = "DEFAULT" if config.pretrain else None

        if config.architecture == DetectorArchitecture.MOBILENET_V3:
            return fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        elif config.architecture == DetectorArchitecture.RESNET_50:
            return fasterrcnn_resnet50_fpn_v2(weights=weights)

        raise ValueError(f"Unsupported model architecture: {config.architecture}")
