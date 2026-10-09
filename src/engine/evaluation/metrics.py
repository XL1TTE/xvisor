from __future__ import annotations

import torch


def compute_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Computes pairwise Intersection-over-Union (IoU) between two sets of boxes.

    Args:
        boxes_a: FloatTensor [N, 4] in [x1, y1, x2, y2]
        boxes_b: FloatTensor [M, 4] in [x1, y1, x2, y2]

    Returns:
        FloatTensor [N, M] with IoU overlap values in range [0.0, 1.0]
    """
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return torch.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=torch.float32)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (
        boxes_a[:, 3] - boxes_a[:, 1]
    ).clamp(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (
        boxes_b[:, 3] - boxes_b[:, 1]
    ).clamp(min=0)

    # Intersections
    lt = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])  # [N, M, 2]
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-8)


def calculate_ap_for_iou(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    iou_threshold: float,
) -> tuple[float, float, float]:
    """Calculates Average Precision, Precision, and Recall at a specific IoU threshold.

    Returns:
        (ap, precision_at_all, recall_at_all)
    """
    total_gt = sum(t["boxes"].shape[0] for t in targets)
    if total_gt == 0:
        return 0.0, 0.0, 0.0

    all_scores: list[float] = []
    all_matched: list[int] = []

    for pred, target in zip(predictions, targets):
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        gt_boxes = target["boxes"]

        num_preds = pred_boxes.shape[0]
        if num_preds == 0:
            continue

        for s in pred_scores:
            all_scores.append(s.item())

        if gt_boxes.shape[0] == 0:
            all_matched.extend([0] * num_preds)
            continue

        ious = compute_iou(pred_boxes, gt_boxes)  # [num_preds, num_gts]
        matched_gt = set()

        # Sort predictions by score descending
        sorted_indices = torch.argsort(pred_scores, descending=True)
        is_tp = [0] * num_preds

        for p_idx in sorted_indices:
            best_iou, best_gt_idx = torch.max(ious[p_idx], dim=0)
            best_gt = best_gt_idx.item()

            if best_iou.item() >= iou_threshold and best_gt not in matched_gt:
                is_tp[p_idx] = 1
                matched_gt.add(best_gt)
            else:
                is_tp[p_idx] = 0

        all_matched.extend(is_tp)

    if not all_scores:
        return 0.0, 0.0, 0.0

    # Sort all predictions across the entire dataset by confidence score descending
    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
    matched_tensor = torch.tensor(all_matched, dtype=torch.float32)

    sort_order = torch.argsort(scores_tensor, descending=True)
    matched_sorted = matched_tensor[sort_order]

    tp_cumsum = torch.cumsum(matched_sorted, dim=0)
    fp_cumsum = torch.cumsum(1.0 - matched_sorted, dim=0)

    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # 11-point interpolation for Average Precision (standard VOC/COCO style)
    ap = 0.0
    for t in torch.linspace(0.0, 1.0, 11):
        prec_at_recall = precisions[recalls >= t]
        if prec_at_recall.numel() > 0:
            ap += torch.max(prec_at_recall).item()
    ap /= 11.0

    final_precision = precisions[-1].item() if precisions.numel() > 0 else 0.0
    final_recall = recalls[-1].item() if recalls.numel() > 0 else 0.0

    return ap, final_precision, final_recall
