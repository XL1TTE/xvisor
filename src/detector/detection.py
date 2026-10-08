from dataclasses import dataclass
from torch import Tensor


@dataclass(frozen=True)
class Detection:
    boxes: Tensor
    scores: Tensor
    labels: Tensor
