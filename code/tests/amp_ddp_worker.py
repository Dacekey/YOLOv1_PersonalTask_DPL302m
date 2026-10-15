"""Inject an asymmetric overflow to verify rank-coordinated optimizer rejection."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
import distributed_utils as distributed
from precision import optimizer_step


def main():
    device = distributed.initialize("cpu")
    torch.set_num_threads(1)
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    scaler = torch.amp.GradScaler("cpu", init_scale=128)
    scaler.scale(model(torch.ones(1, 1)).sum()).backward()
    if distributed.rank() == 1:
        model.weight.grad.fill_(float("inf"))
    result = optimizer_step(model, optimizer, scaler, 10, device)
    assert model.weight.item() == 0
    state = {"overflow_skipped": not result["applied"], "scale": scaler.get_scale()}
    scaler.scale(model(torch.ones(1, 1)).sum()).backward()
    state["recovered"] = optimizer_step(model, optimizer, scaler, 10, device)["applied"]
    model.weight.grad = torch.tensor([[float("nan") if distributed.rank() == 1 else 1.]])
    try:
        optimizer_step(model, optimizer, torch.amp.GradScaler("cpu", enabled=False), 10, device)
        state["fp32_rejected"] = False
    except FloatingPointError:
        state["fp32_rejected"] = True
    Path(sys.argv[1], f"rank_{distributed.rank()}.json").write_text(json.dumps(state))
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
