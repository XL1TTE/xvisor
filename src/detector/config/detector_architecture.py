from enum import Enum


class DetectorArchitecture(str, Enum):
    MOBILENET_V3 = "mobilenet_v3"
    RESNET_50 = "resnet_50"
