import json
from pathlib import Path

from data.annotations import (
    classify_annotation_file,
    parse_annotations,
    CocoJsonAnnotation,
    MotTxtAnnotation,
)


def test_parse_coco_json(tmp_path: Path) -> None:
    coco_data = {
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
        "images": [
            {"id": 10, "file_name": "frame_0.jpg"},
            {"id": 20, "file_name": "frame_1.jpg"},
        ],
        "annotations": [
            # Human in image 10: [x=10, y=20, w=30, h=40] -> [x1=10, y1=20, x2=40, y2=60]
            {"id": 1, "image_id": 10, "category_id": 1, "bbox": [10.0, 20.0, 30.0, 40.0]},
            # Car in image 10 (should be filtered out)
            {"id": 2, "image_id": 10, "category_id": 2, "bbox": [50.0, 50.0, 100.0, 100.0]},
        ],
    }

    json_file = tmp_path / "annotations.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(coco_data, f)

    fmt = classify_annotation_file(json_file)
    assert isinstance(fmt, CocoJsonAnnotation)

    annotations = parse_annotations(fmt)
    assert len(annotations) == 2

    # Frame 0 (image id 10)
    frame0 = annotations[0]
    assert len(frame0.objects) == 1
    obj = frame0.objects[0]
    assert obj.category_id == 1
    assert obj.box.x1 == 10.0
    assert obj.box.y1 == 20.0
    assert obj.box.x2 == 40.0
    assert obj.box.y2 == 60.0

    # Frame 1 (image id 20 - has no people)
    frame1 = annotations[1]
    assert len(frame1.objects) == 0


def test_parse_mot_txt(tmp_path: Path) -> None:
    # MOT: <frame>, <id>, <x>, <y>, <w>, <h>, <conf>, <class>, <vis>
    mot_lines = [
        "1, 101, 15.0, 25.0, 50.0, 100.0, 1, 1, 1.0\n",
        "1, 102, 70.0, 80.0, 30.0, 60.0, 1, 1, 0.9\n",
        "2, 101, 18.0, 28.0, 50.0, 100.0, 1, 1, 1.0\n",
    ]

    txt_file = tmp_path / "gt.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.writelines(mot_lines)

    fmt = classify_annotation_file(txt_file)
    assert isinstance(fmt, MotTxtAnnotation)

    annotations = parse_annotations(fmt)
    assert len(annotations) == 2

    # Frame 0 (from frame 1 in file)
    frame0 = annotations[0]
    assert len(frame0.objects) == 2
    assert frame0.objects[0].track_id == 101
    assert frame0.objects[0].box.x1 == 15.0
    assert frame0.objects[0].box.x2 == 65.0  # 15 + 50

    # Frame 1 (from frame 2 in file)
    frame1 = annotations[1]
    assert len(frame1.objects) == 1
    assert frame1.objects[0].track_id == 101
