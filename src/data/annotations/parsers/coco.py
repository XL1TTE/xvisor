from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List

from ..types import (
    CocoJsonAnnotation,
    FrameAnnotation,
    ObjectAnnotation,
    BoundingBox,
)


def parse_coco_json(
    annotation: CocoJsonAnnotation,
    target_category_name: str = "person",
) -> dict[int, FrameAnnotation]:
    with open(annotation.file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Identify category_id for human/person
    categories = data.get("categories", [])
    person_cat_ids = {
        cat["id"]
        for cat in categories
        if cat.get("name", "").lower() == target_category_name.lower()
    }

    # If no categories defined, fallback to ID 1 as standard COCO person
    if not person_cat_ids:
        person_cat_ids = {1}

    # 2. Map image_id to 0-based frame index based on sorted images list
    images = sorted(data.get("images", []), key=lambda img: img.get("id", 0))
    image_id_to_frame_idx: dict[int, int] = {
        img["id"]: idx for idx, img in enumerate(images)
    }

    # 3. Collect objects grouped by image_id
    grouped_objects: Dict[int, List[ObjectAnnotation]] = defaultdict(list)
    for ann in data.get("annotations", []):
        cat_id = ann.get("category_id")
        if cat_id not in person_cat_ids:
            continue

        image_id = ann.get("image_id")
        if image_id not in image_id_to_frame_idx:
            continue

        # COCO bbox format: [x, y, width, height]
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            continue

        x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        if w <= 0 or h <= 0:
            continue

        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        grouped_objects[image_id].append(
            ObjectAnnotation(
                box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                category_id=1,
                track_id=ann.get("track_id", None),
            )
        )

    # 4. Construct FrameAnnotation for every image in the dataset
    result: dict[int, FrameAnnotation] = {}
    for img in images:
        img_id = img["id"]
        frame_idx = image_id_to_frame_idx[img_id]
        objects = tuple(grouped_objects.get(img_id, []))
        result[frame_idx] = FrameAnnotation(frame_index=frame_idx, objects=objects)

    return result
