import json
import zipfile
from pathlib import Path

import pytest
import torch

from dataset import encode_target, read_annotation
from inference_demo import (match_detections, operating_metrics, resolve_checkpoint,
                            run_demo, select_examples)
from test_detection import make_voc


def test_match_duplicate_difficult_and_threshold():
    annotation = {"boxes": torch.tensor([[0., 0., .2, .2], [.4, .4, .6, .6]]),
        "labels": torch.tensor([14, 14]), "difficult": torch.tensor([False, True])}
    predictions = [{"class_id": 14, "score": .9, "box": [0., 0., .2, .2]},
        {"class_id": 14, "score": .8, "box": [0., 0., .2, .2]},
        {"class_id": 14, "score": .7, "box": [.4, .4, .6, .6]},
        {"class_id": 1, "score": .6, "box": [0., 0., .2, .2]},
        {"class_id": 14, "score": .1, "box": [0., 0., .2, .2]}]
    matches = match_detections(predictions, annotation, .25)
    assert [m["status"] for m in matches] == ["TP", "FP", "ignored_difficult", "FP"]
    metrics, images, _ = operating_metrics({"a": predictions}, {"a": annotation}, .25)
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 2, 0)
    assert metrics["precision"] == pytest.approx(1 / 3)
    assert metrics["recall"] == 1 and metrics["f1"] == pytest.approx(.5)
    assert metrics["mean_tp_iou"] == 1


def test_selection_prefers_correct_diverse_examples_and_fallback():
    def row(name, tp, precision, classes):
        return {"image_id": name, "detections": 1, "tp": tp, "precision": precision,
                "mean_tp_iou": .9 if tp else None, "max_score": .8, "tp_classes": classes}
    rows = {"a": row("a", 1, 1, [0]), "b": row("b", 1, .9, [0]),
            "c": row("c", 1, .8, [14]), "d": row("d", 0, 0, [])}
    assert [r["image_id"] for r in select_examples(rows, 3)] == ["a", "c", "b"]
    assert select_examples({}, 3) == []
    assert len(select_examples({"d": rows["d"]}, 3)) == 1


def test_checkpoint_resolution_only_extracts_best_and_rejects_ambiguity(tmp_path):
    inputs = tmp_path / "input"; inputs.mkdir()
    archive_path = inputs / "misleading_smoke_name.zip"
    with zipfile.ZipFile(archive_path, "w") as z:
        z.writestr("run/checkpoints/best.pt", b"test weights")
        z.writestr("run/checkpoints/last.pt", b"do not extract")
    checkpoint, origin = resolve_checkpoint("", inputs, tmp_path / "cache")
    assert checkpoint.read_bytes() == b"test weights"
    assert list(checkpoint.parent.iterdir()) == [checkpoint]
    assert origin["kind"] == "training_zip"
    (inputs / "a").mkdir(); (inputs / "b").mkdir()
    (inputs / "a/best.pt").write_bytes(b"a"); (inputs / "b/best.pt").write_bytes(b"b")
    with pytest.raises(ValueError, match="Multiple"):
        resolve_checkpoint("", inputs, tmp_path / "cache")


def test_inference_export_and_notebook(tmp_path, monkeypatch):
    import inference_demo
    root = make_voc(tmp_path / "dataset")
    target, _, _ = encode_target(read_annotation(root / "Annotations/000002.xml"))
    class CorrectDetector(torch.nn.Module):
        def forward(self, images):
            out = images.new_zeros((len(images), 7, 7, 30))
            out[..., :20] = target[..., :20].to(images.device)
            cells = torch.where(target[..., 20] > 0)
            for row, col in zip(*cells):
                out[:, row, col, 20:22] = target[row, col, 21:23]
                out[:, row, col, 22:24] = target[row, col, 23:25].sqrt()
                out[:, row, col, 24] = .9
            return out
    checkpoint = tmp_path / "best.pt"; checkpoint.write_bytes(b"fixture")
    metadata = {"epoch": 49, "config": {"epochs": 50}, "split_ids": {"train": ["000001"]}}
    monkeypatch.setattr(inference_demo, "load_detector", lambda path, device: (CorrectDetector(), metadata))
    output = tmp_path / "demo"; archive = tmp_path / "demo.zip"
    manifest = run_demo(root, checkpoint, output, archive, device="cpu", count=3)
    assert manifest["status"] == "completed" and manifest["selected_count"] == 1
    assert manifest["selected_image_ids"] == ["000002"]
    assert json.loads((output / "metrics.json").read_text())["map50"] == 1
    assert json.loads((output / "operating_point_metrics.json").read_text())["precision"] == 1
    assert (output / "examples/000002_comparison.jpg").is_file()
    assert (output / "examples/contact_sheet.jpg").is_file()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert not any(name.endswith(".pt") for name in z.namelist())
    notebook = Path(__file__).resolve().parents[1] / "notebooks/kaggle_inference_yolov1.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    for cell in cells:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "inference-cell", "exec")
    all_code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    assert "--limit" in all_code  # appears only in the explicit explanatory comment
    assert "torch.distributed.run" not in all_code
    assert 'CODE = REPO / "code"' in all_code
