import json
import subprocess
import sys
from pathlib import Path

from PIL import Image
import torch

from test_detection import make_voc


def test_two_process_ddp_accumulation_and_resume(tmp_path):
    root = make_voc(tmp_path / "dataset")
    # Five train images => one padded duplicate across two ranks. Three local
    # micro-batches exercise no_sync accumulation and the final partial group.
    train_ids = ["000001"]
    xml = (root / "Annotations/000001.xml").read_text()
    for i in range(3, 7):
        image_id = f"{i:06d}"; train_ids.append(image_id)
        Image.new("RGB", (80, 60), (30 * i, 20 * i, 10 * i)).save(root / "JPEGImages" / f"{image_id}.jpg")
        (root / "Annotations" / f"{image_id}.xml").write_text(xml)
    (root / "ImageSets/Main/train.txt").write_text("\n".join(train_ids) + "\n")
    worker = Path(__file__).parent / "ddp_worker.py"
    def run(name, extra=()):
        output = tmp_path / name
        subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                        "--nnodes=1", "--nproc_per_node=2", str(worker), "--data", str(root),
                        "--output", str(output), *map(str, extra)],
                       check=True, capture_output=True, text=True, timeout=120)
        return output
    full = run("full")
    partial = run("partial", ["--interrupt"])
    resumed = run("resumed", ["--resume", partial / "checkpoints/last.pt"])
    expected = torch.load(full / "checkpoints/last.pt", weights_only=True)
    actual = torch.load(resumed / "checkpoints/last.pt", weights_only=True)
    assert expected["world_size"] == 2 and len(actual["rank_states"]) == 2
    assert actual["global_step"] == 4
    for key in expected["model"]:
        assert not key.startswith("module.")
        assert torch.equal(expected["model"][key], actual["model"][key]), key
    assert not torch.equal(actual["rank_states"][0]["rng"]["torch"],
                           actual["rank_states"][1]["rng"]["torch"])
    manifest = json.loads((full / "run_manifest.json").read_text())
    assert manifest["world_size"] == 2 and manifest["effective_batch_size"] == 4
    assert manifest["train_padding_duplicates_per_epoch"] == 1
    assert manifest["status"] == "completed"
    assert json.loads((full / "worker_0_finished.json").read_text())["validation_calls"] == 3
    assert json.loads((full / "worker_1_finished.json").read_text())["validation_calls"] == 0
    assert (tmp_path / "full.zip").is_file()


def test_kaggle_notebook_repo_and_paths():
    code = Path(__file__).resolve().parents[1]
    notebook = json.loads((code / "notebooks/kaggle_train_yolov1.ipynb").read_text())
    cells = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert 'REPO_URL = "https://github.com/Dacekey/YOLOv1_PersonalTask_DPL302m.git"' in cells
    assert 'PRETRAINED_PATH = "code/pretrain/extraction.conv.weights"' in cells
    assert 'CODE = REPO / "code"' in cells and 'PRETRAIN = REPO / PRETRAIN' in cells
    assert '"--nproc_per_node=2"' in cells and 'torch.cuda.device_count() == 2' in cells
    assert (code.parent / "code/pretrain/extraction.conv.weights").is_file()
    for mode in ("smoke", "overfit", "voc2007"):
        config = json.loads((code / "configs" / f"{mode}_ddp.json").read_text())
        assert config["expected_world_size"] == 2


def test_two_process_spawn_dataloader(tmp_path):
    root = make_voc(tmp_path / "dataset")
    worker = Path(__file__).parent / "ddp_worker.py"
    output = tmp_path / "spawn_run"
    subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                    "--nnodes=1", "--nproc_per_node=2", str(worker), "--data", str(root),
                    "--output", str(output), "--workers", "1", "--interrupt"],
                   check=True, capture_output=True, text=True, timeout=120)
    checkpoint = torch.load(output / "checkpoints/last.pt", weights_only=True)
    assert checkpoint["epoch"] == 0 and checkpoint["world_size"] == 2
    assert len(checkpoint["rank_states"]) == 2
