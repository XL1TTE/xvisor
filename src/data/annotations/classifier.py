from __future__ import annotations

from pathlib import Path
from .types import AnnotationFormat, CocoJsonAnnotation, MotTxtAnnotation


def classify_annotation_file(file_path: Path) -> AnnotationFormat:
    if not file_path.exists():
        raise FileNotFoundError(f"Annotation file does not exist: '{file_path}'")

    match file_path.suffix.lower():
        case ".json":
            return CocoJsonAnnotation(file_path=file_path)
        case ".txt" | ".csv":
            return MotTxtAnnotation(file_path=file_path)
        case _:
            raise ValueError(
                f"Unsupported annotation file format: '{file_path.suffix}' for file '{file_path}'"
            )
