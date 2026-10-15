from pathlib import Path
import pytest
import torch
from pretrained import EXPECTED_SHA256, load_pretrained
from yolov1 import YOLOv1


def test_supplied_weights_prefix_and_forward():
    path = Path(__file__).resolve().parents[1] / "pretrain/extraction.conv.weights"
    if not path.is_file():
        pytest.skip("Attach the verified extraction.conv.weights to run this test")
    torch.set_num_threads(4)
    backbone = YOLOv1._make_backbone()
    untouched = backbone.conv17.weight.detach().clone()
    report = load_pretrained(backbone, path)
    assert report["sha256"] == EXPECTED_SHA256 and len(report["loaded_layers"]) == 16
    assert report["loaded_parameters"] == 11914304
    assert torch.equal(untouched, backbone.conv17.weight)
    with torch.inference_mode():
        x = torch.rand(1, 3, 448, 448)
        for name, layer in backbone.named_children():
            x = layer(x)
            if name == "relu16":
                break
    assert x.shape == (1, 1024, 28, 28) and torch.isfinite(x).all()


def test_bad_weights_rejected_before_mutation(tmp_path):
    path = tmp_path / "invalid.weights"; path.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="bytes"):
        load_pretrained(YOLOv1._make_backbone(), path)
