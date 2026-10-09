from pathlib import Path
import pytest
import torch

from reid import ReIDModel
from export import export_reid_to_onnx, verify_onnx_model


def test_reid_model_forward_pass() -> None:
    model = ReIDModel(pretrained=False)
    model.eval()

    dummy_batch = torch.randn(2, 3, 128, 64)
    with torch.no_grad():
        output = model(dummy_batch)

    assert output.shape == (2, 576)
    # Check L2 normalization: norm across dimension 1 should be 1.0
    norms = torch.norm(output, p=2, dim=1)
    for n in norms:
        assert abs(n.item() - 1.0) < 1e-5


def test_export_reid_to_onnx(tmp_path: Path) -> None:
    model = ReIDModel(pretrained=False)
    output_file = tmp_path / "test_reid.onnx"

    exported = export_reid_to_onnx(model, output_file)
    assert exported.exists()
    assert exported.stat().st_size > 0

    assert verify_onnx_model(exported)
