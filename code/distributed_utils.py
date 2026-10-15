"""torchrun initialization and small synchronized training control messages."""
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def initialize(device):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        if device.startswith("cuda"):
            if torch.cuda.device_count() < world_size:
                raise RuntimeError(f"torchrun requests {world_size} GPUs, found {torch.cuda.device_count()}")
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        dist.init_process_group(backend="nccl" if device.startswith("cuda") else "gloo",
                                timeout=timedelta(hours=2))
    return device


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def broadcast(value):
    if world_size() == 1:
        return value
    payload = [value if rank() == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def any_rank(value, device):
    flag = torch.tensor(int(bool(value)), device=device)
    if world_size() > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def gather_states(state):
    if world_size() == 1:
        return [state]
    states = [None] * world_size()
    dist.all_gather_object(states, state)
    return states
