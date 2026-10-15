import json
import math
import zipfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from boxes import box_iou, decode_predictions, nms
from dataset import VOCDataset, encode_target, find_voc_root, read_annotation
from evaluate import voc_metrics
from loss import YOLOv1Loss


def annotation(boxes, labels=None, difficult=None):
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels or [0] * len(boxes)),
            "difficult": torch.tensor(difficult or [False] * len(boxes), dtype=torch.bool)}


def make_voc(root):
    root = root / "VOCdevkit2007/VOC2007"
    for name in ("JPEGImages", "Annotations", "ImageSets/Main"):
        (root / name).mkdir(parents=True)
    for split, image_id in (("train", "000001"), ("val", "000002")):
        (root / "ImageSets/Main" / f"{split}.txt").write_text(image_id + "\n")
        Image.new("RGB", (80, 60), "gray").save(root / "JPEGImages" / f"{image_id}.jpg")
        (root / "Annotations" / f"{image_id}.xml").write_text('''<annotation>
          <size><width>80</width><height>60</height></size><object><name>person</name>
          <difficult>0</difficult><bndbox><xmin>9</xmin><ymin>7</ymin><xmax>40</xmax>
          <ymax>42</ymax></bndbox></object></annotation>''')
    return root


def test_target_collision_and_difficult():
    ann = annotation([[.1, .1, .3, .3], [.05, .05, .35, .35], [.7, .7, .8, .8]],
                     [1, 2, 3], [False, False, True])
    grid, ignored, collisions = encode_target(ann)
    assert collisions == 1 and grid[..., 20].sum() == 1
    assert grid[1, 1, 2] == 1 and ignored[5, 5]
    assert len(ann["boxes"]) == 3  # Evaluation GT is never dropped.


def test_perfect_loss_decode_and_gradients():
    target, ignored, _ = encode_target(annotation([[.1, .1, .3, .3]], [14]))
    prediction = torch.zeros(1, 7, 7, 30)
    prediction[..., :20] = target[..., :20]
    prediction[0, 1, 1, 20:22] = target[1, 1, 21:23]
    prediction[0, 1, 1, 22:24] = target[1, 1, 23:25].sqrt()
    prediction[0, 1, 1, 24] = 1
    prediction.requires_grad_()
    loss, terms = YOLOv1Loss()(prediction, target[None], ignored[None])
    assert float(loss.detach()) < 1e-10
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    decoded = decode_predictions(prediction, .1)[0]
    assert len(decoded) == 1 and decoded[0]["class_id"] == 14
    assert decoded[0]["box"] == pytest.approx([.1, .1, .3, .3], abs=1e-6)


def test_empty_loss_and_difficult_ignore():
    pred = torch.zeros(1, 7, 7, 30, requires_grad=True)
    target = torch.zeros(1, 7, 7, 25)
    value, _ = YOLOv1Loss()(pred, target)
    value.backward()
    assert value == 0 and torch.isfinite(pred.grad).all()
    p = pred.detach().clone(); p[0, 0, 0, 24] = 2; p[0, 0, 0, 29] = 3
    ignore = torch.zeros(1, 7, 7, dtype=torch.bool); ignore[0, 0, 0] = True
    assert YOLOv1Loss()(p, target, ignore)[0] == 0
    assert YOLOv1Loss()(p, target)[0] == pytest.approx(6.5)


def test_responsible_and_noobj_policy():
    target, ignored, _ = encode_target(annotation([[.1, .1, .3, .3]]))
    p = torch.zeros(1, 7, 7, 30)
    p[0, 1, 1, 20:22] = target[1, 1, 21:23]
    p[0, 1, 1, 22:24] = target[1, 1, 23:25].sqrt()
    p[0, 1, 1, 29] = 2  # Non-responsible box is trained toward zero confidence.
    loss, terms = YOLOv1Loss()(p.requires_grad_(), target[None], ignored[None])
    assert terms["no_object"] == pytest.approx(2)
    loss.backward(); assert p.grad[0, 1, 1, 29] == pytest.approx(2)


def test_iou_nms_and_negative_scores():
    boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.], [.7, .7, .9, .9]])
    assert box_iou(boxes[0], boxes[1]) == 1
    assert nms(boxes, torch.tensor([.9, .8, .7])).tolist() == [0, 2]
    p = torch.zeros(1, 7, 7, 30)
    p[0, 0, 0, 0] = -1; p[0, 0, 0, 22:24] = .3; p[0, 0, 0, 24] = -1
    assert decode_predictions(p, .1) == [[]]


def test_voc_metric_difficult_duplicate_and_missing_class():
    ann = {"a": annotation([[0, 0, .2, .2], [.4, .4, .6, .6]], [0, 0], [True, False]),
           "b": annotation([[.4, .4, .6, .6]])}
    pred = {"a": [{"class_id": 0, "score": .95, "box": [0, 0, .2, .2]},
                   {"class_id": 0, "score": .9, "box": [.4, .4, .6, .6]},
                   {"class_id": 0, "score": .8, "box": [.4, .4, .6, .6]}],
            "b": [{"class_id": 0, "score": .7, "box": [.4, .4, .6, .6]}]}
    result = voc_metrics(pred, ann)
    row = result["per_class"][0]
    assert row["tp"] == 2 and row["fp"] == 1 and row["gt_objects"] == 2
    assert row["ap"] == pytest.approx((6 + 5 * 2 / 3) / 11)
    assert result["classes_with_gt"] == 1 and result["per_class"][1]["ap"] is None


def test_dataset_xml_and_split(tmp_path):
    root = make_voc(tmp_path)
    assert find_voc_root(tmp_path) == root
    ann = read_annotation(root / "Annotations/000001.xml")
    assert ann["boxes"][0].tolist() == pytest.approx([.1, .1, .5, .7])
    images, target = VOCDataset(root)[0]
    assert images.shape == (3, 448, 448) and target["image_id"] == "000001"
    with pytest.raises(FileNotFoundError):
        VOCDataset(root, "test")


def test_train_resume_export_with_small_network(tmp_path, monkeypatch):
    """Exercise the real orchestration using a small stand-in; no claim of full-model training."""
    import train
    from run_utils import atomic_save
    root = make_voc(tmp_path / "dataset")
    class SmallDetector(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.fc = torch.nn.Linear(3, 1470)
        def forward(self, images):
            return self.fc(images.mean((2, 3))).reshape(-1, 7, 7, 30)
    monkeypatch.setattr(train, "build_model", lambda config, device, pretrained_path: (SmallDetector().to(device), None))
    def load(path, device):
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        model = SmallDetector().to(device); model.load_state_dict(checkpoint["model"])
        return model, checkpoint
    monkeypatch.setattr(train, "load_detector", load)
    config = {"epochs": 2, "batch_size": 1, "accumulation": 2, "learning_rate": .001,
              "seed": 42, "num_workers": 0, "augment": False, "amp": False,
              "require_pretrained": False, "example_count": 1, "max_detections": 10,
              "export_reserve_seconds": 0, "log_every": 1}
    out = tmp_path / "full"
    train.train_run(config, root, out, device="cpu", zip_path=tmp_path / "full.zip")
    checkpoint = torch.load(out / "checkpoints/last.pt", weights_only=True)
    assert checkpoint["epoch"] == 1 and checkpoint["global_step"] == 2
    assert checkpoint["optimizer"]["state"]
    # Create an epoch-one resume point by rerunning with a signal flag after the first eval.
    original_eval = train.evaluate_model
    import signal
    calls = [0]
    def interrupt_after_validation(*args, **kwargs):
        result = original_eval(*args, **kwargs); calls[0] += 1
        if calls[0] == 1:
            signal.raise_signal(signal.SIGTERM)
        return result
    monkeypatch.setattr(train, "evaluate_model", interrupt_after_validation)
    partial = tmp_path / "partial"
    train.train_run(config, root, partial, device="cpu")
    monkeypatch.setattr(train, "evaluate_model", original_eval)
    resumed = tmp_path / "resumed"
    train.train_run(config, root, resumed, resume=partial / "checkpoints/last.pt", device="cpu")
    restored = torch.load(resumed / "checkpoints/last.pt", weights_only=True)
    for name, value in checkpoint["model"].items():
        assert torch.equal(value, restored["model"][name]), name
    with zipfile.ZipFile(tmp_path / "full.zip") as archive:
        assert archive.testzip() is None
        assert "full/checkpoints/last.pt" in archive.namelist()
        assert "full/metrics.json" in archive.namelist()
        assert "full/examples/000002.jpg" in archive.namelist()


def test_notebook_cells_compile():
    notebook = Path(__file__).resolve().parents[1] / "notebooks/kaggle_train_yolov1.ipynb"
    for cell in json.loads(notebook.read_text())["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "notebook-cell", "exec")
