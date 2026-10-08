from __future__ import annotations

from .types import (
    AnnotationFormat,
    CocoJsonAnnotation,
    MotTxtAnnotation,
    FrameAnnotation,
)
from .parsers.coco import parse_coco_json
from .parsers.mot import parse_mot_txt


def parse_annotations(format_meta: AnnotationFormat) -> dict[int, FrameAnnotation]:
    match format_meta:
        case CocoJsonAnnotation() as coco:
            return parse_coco_json(coco)
        case MotTxtAnnotation() as mot:
            return parse_mot_txt(mot)
