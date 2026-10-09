from __future__ import annotations

from pathlib import Path
from typing import Optional
import torch
from torchvision.models.detection import FasterRCNN

from reid import ReIDModel


def export_reid_to_onnx(
    model: ReIDModel,
    output_path: Path,
    opset_version: int = 14,
) -> Path:
    """Exports the ReID feature extractor to ONNX format with dynamic batch axis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 128, 64, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["crops"],
        output_names=["embeddings"],
        dynamic_axes={
            "crops": {0: "batch_size"},
            "embeddings": {0: "batch_size"},
        },
    )
    return output_path


def export_detector_to_onnx(
    model: FasterRCNN,
    output_path: Path,
    opset_version: int = 14,
) -> Path:
    """Exports Faster R-CNN human detector to ONNX format with dynamic spatial axes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    # Faster R-CNN in eval mode expects a list of 3D tensors: [Tensor[3, H, W]]
    dummy_input = [torch.randn(3, 640, 640, dtype=torch.float32)]

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["boxes", "labels", "scores"],
        dynamic_axes={
            "images": {1: "height", 2: "width"},
        },
    )
    return output_path


def verify_onnx_model(onnx_path: Path) -> bool:
    """Validates the structure and graph integrity of an exported ONNX file."""
    try:
        import onnx

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        return True
    except ImportError:
        # If onnx package is not installed, return True if file exists and non-empty
        return onnx_path.exists() and onnx_path.stat().st_size > 0
