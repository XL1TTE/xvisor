from __future__ import annotations

import csv
from collections import defaultdict
from typing import Dict, List

from ..types import (
    MotTxtAnnotation,
    FrameAnnotation,
    ObjectAnnotation,
    BoundingBox,
)


def parse_mot_txt(annotation: MotTxtAnnotation) -> dict[int, FrameAnnotation]:
    """Parses MOTChallenge ground truth file.

    Format per line:
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
    """
    grouped_objects: Dict[int, List[ObjectAnnotation]] = defaultdict(list)
    all_frames: set[int] = set()

    with open(annotation.file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line in reader:
            if not line or len(line) < 6:
                continue

            # Strip whitespace
            parts = [p.strip() for p in line]

            # 1-based frame number in MOT format
            frame_num = int(float(parts[0]))
            # Convert to 0-based frame index
            frame_idx = max(0, frame_num - 1)
            all_frames.add(frame_idx)

            track_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])

            if w <= 0 or h <= 0:
                continue

            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            grouped_objects[frame_idx].append(
                ObjectAnnotation(
                    box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    category_id=1,
                    track_id=track_id,
                )
            )

    result: dict[int, FrameAnnotation] = {}
    if not all_frames:
        return result

    max_frame = max(all_frames)
    for idx in range(max_frame + 1):
        objects = tuple(grouped_objects.get(idx, []))
        result[idx] = FrameAnnotation(frame_index=idx, objects=objects)

    return result
