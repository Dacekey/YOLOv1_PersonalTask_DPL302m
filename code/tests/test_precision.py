import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from dataset import encode_target
from loss import YOLOv1Loss
from precision import optimizer_step


def test_fp16_overflow_backs_off_without_changing_parameters():
    model = torch.nn.Linear(3, 1470)
    torch.nn.init.zeros_(model.weight); torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    scaler = torch.amp.GradScaler("cpu", init_scale=65536)
    target, ignored, _ = encode_target({"boxes": torch.tensor([[.1, .1, .3, .3]]),
        "labels": torch.tensor([14]), "difficult": torch.tensor([False])})
    skips = 0
    for _ in range(12):
        before = [p.detach().clone() for p in model.parameters()]
        with torch.autocast("cpu", dtype=torch.float16):
            prediction = model(torch.ones(1, 3)).reshape(1, 7, 7, 30)
            loss, _ = YOLOv1Loss()(prediction, target[None], ignored[None])
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        result = optimizer_step(model, optimizer, scaler, 10, torch.device("cpu"))
        if result["applied"]:
            break
        skips += 1
        assert result["scale_after"] == result["scale_before"] / 2
        assert all(torch.equal(p, old) for p, old in zip(model.parameters(), before))
        assert all(p.grad is None for p in model.parameters())
    assert skips > 0 and result["applied"]
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(p, old) for p, old in zip(model.parameters(), before))


def test_finite_gradient_clipping_and_fp32_rejection():
    model = torch.nn.Linear(2, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    model(torch.tensor([[3., 4.]])).sum().backward()
    result = optimizer_step(model, optimizer, scaler, 2, torch.device("cpu"))
    assert result["applied"] and result["grad_norm"] == pytest.approx(5)
    assert model.weight.detach().tolist()[0] == pytest.approx([-1.2, -1.6], abs=1e-6)
    before = model.weight.detach().clone()
    model.weight.grad = torch.full_like(model.weight, float("nan"))
    with pytest.raises(FloatingPointError, match="FP32"):
        optimizer_step(model, optimizer, scaler, 2, torch.device("cpu"))
    assert torch.equal(model.weight, before)


def test_amp_initial_scale_128_handles_yolo_loss():
    model = torch.nn.Linear(3, 1470)
    torch.nn.init.zeros_(model.weight); torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    scaler = torch.amp.GradScaler("cpu", init_scale=128)
    target = torch.zeros(1, 7, 7, 25)
    target[0, 1, 1, 14] = 1; target[0, 1, 1, 20:25] = torch.tensor([1., .4, .4, .2, .2])
    with torch.autocast("cpu", dtype=torch.float16):
        prediction = model(torch.ones(1, 3)).reshape(1, 7, 7, 30)
        loss, _ = YOLOv1Loss()(prediction, target)
    scaler.scale(loss).backward()
    assert optimizer_step(model, optimizer, scaler, 10, torch.device("cpu"))["applied"]


def test_ddp_overflow_is_skipped_on_both_ranks(tmp_path):
    worker = Path(__file__).parent / "amp_ddp_worker.py"
    subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                    "--nnodes=1", "--nproc_per_node=2", str(worker), str(tmp_path)],
                   check=True, capture_output=True, text=True, timeout=90)
    states = [json.loads((tmp_path / f"rank_{r}.json").read_text()) for r in range(2)]
    assert states[0] == states[1]
    assert states[0]["overflow_skipped"] and states[0]["recovered"]
    assert states[0]["fp32_rejected"] and states[0]["scale"] == 64
