from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN

from data.dataset import TrackingDataset
from data.collation import collate_tracking_batch
from .types import EvaluationConfig, EvaluationMetrics
from .metrics import calculate_ap_for_iou


class DetectorEvaluator:
    """Evaluates a trained detector over a test dataset to compute detection metrics."""

    def __init__(
        self,
        model: FasterRCNN,
        device: torch.device,
        config: EvaluationConfig = EvaluationConfig(),
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.config = config

    @torch.no_grad()
    def evaluate(self, dataset: TrackingDataset) -> EvaluationMetrics:
        self.model.eval()

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_tracking_batch,
        )

        all_filtered_predictions: list[dict[str, torch.Tensor]] = []
        all_targets: list[dict[str, torch.Tensor]] = []

        total_preds = 0
        total_gts = 0

        for images, targets in dataloader:
            images = [img.to(self.device) for img in images]
            raw_predictions = self.model(images)

            for pred, target in zip(raw_predictions, targets):
                # Filter by confidence threshold & human class (label == 1)
                scores = pred["scores"].cpu()
                boxes = pred["boxes"].cpu()
                labels = pred["labels"].cpu()

                mask = (scores >= self.config.confidence_threshold) & (labels == 1)
                filtered_boxes = boxes[mask]
                filtered_scores = scores[mask]

                total_preds += filtered_boxes.shape[0]
                total_gts += target["boxes"].shape[0]

                all_filtered_predictions.append(
                    {
                        "boxes": filtered_boxes,
                        "scores": filtered_scores,
                    }
                )
                all_targets.append(
                    {
                        "boxes": target["boxes"].cpu(),
                    }
                )

        # 1. Compute AP at IoU 0.50
        ap50, prec, rec = calculate_ap_for_iou(
            all_filtered_predictions,
            all_targets,
            iou_threshold=self.config.iou_threshold,
        )

        # 2. Compute mAP across IoU 0.50:0.95 (step 0.05)
        iou_thresholds = [0.50 + 0.05 * i for i in range(10)]
        aps = [
            calculate_ap_for_iou(all_filtered_predictions, all_targets, iou)[0]
            for iou in iou_thresholds
        ]
        map50_95 = sum(aps) / len(aps)

        return EvaluationMetrics(
            ap50=ap50,
            map50_95=map50_95,
            precision=prec,
            recall=rec,
            total_ground_truth=total_gts,
            total_predictions=total_preds,
        )
