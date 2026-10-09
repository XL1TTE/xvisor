from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationConfig:
    batch_size: int = 2
    confidence_threshold: float = 0.50
    iou_threshold: float = 0.50


@dataclass(frozen=True)
class EvaluationMetrics:
    ap50: float
    map50_95: float
    precision: float
    recall: float
    total_ground_truth: int
    total_predictions: int
