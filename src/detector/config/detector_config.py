from dataclasses import dataclass

from .detector_architecture import DetectorArchitecture


@dataclass(frozen=True)
class DetectorConfig:
    architecture: DetectorArchitecture
    num_classes: int
    pretrain: bool
