"""Shared configuration, reproducibility, and checkpoint helpers."""
import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)


def rng_state():
    np_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [np_state[0], np_state[1].tolist(),
            np_state[2], np_state[3], np_state[4]], "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def atomic_save(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temp)
    os.replace(temp, path)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent, stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.SubprocessError, OSError):
        commit = None
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": str(np.__version__), "git_commit": commit,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None}


def load_detector(checkpoint, device="cpu"):
    from yolov1 import YOLOv1
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if ckpt.get("format") != "personal-yolov1-v1":
        raise ValueError("Not a checkpoint from this implementation")
    with torch.device("meta"):
        model = YOLOv1(dropout=ckpt["config"].get("dropout", 0.5))
    model.load_state_dict(ckpt["model"], strict=True, assign=True)
    return model.to(device), ckpt
