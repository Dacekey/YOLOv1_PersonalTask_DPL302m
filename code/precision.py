"""Gradient clipping with synchronized AMP overflow backoff, before any mutation."""
import math

import torch
import distributed_utils as distributed


def optimizer_step(model, optimizer, scaler, max_norm, device):
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("grad_clip must be finite and positive")
    scaler.unscale_(optimizer)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients:
        raise RuntimeError("No gradients were produced")
    # Per-tensor reductions stay FP32. Only the small vector of norms uses FP64
    # so its sum of squares cannot overflow at ordinary finite FP32 magnitudes.
    norms = torch.stack([torch.linalg.vector_norm(g.detach().float()) for g in gradients])
    norm = torch.linalg.vector_norm(norms.double())
    invalid = distributed.any_rank(not bool(torch.isfinite(norm)), device)
    scale_before = float(scaler.get_scale())
    if invalid:
        if not scaler.is_enabled():
            raise FloatingPointError("Non-finite unscaled gradients in FP32; optimizer update rejected")
        # A peer may have overflowed even if this rank did not. Skip on every
        # rank and explicitly back off through the public API to keep scales equal.
        scale_after = scale_before * scaler.get_backoff_factor()
        if not math.isfinite(scale_after) or scale_after <= 0:
            raise FloatingPointError("AMP loss scale exhausted while backing off non-finite gradients")
        scaler.update(new_scale=scale_after)
        optimizer.zero_grad(set_to_none=True)
        return {"applied": False, "grad_norm": None,
                "scale_before": scale_before, "scale_after": scale_after}
    coefficient = (max_norm / (norm + 1e-6)).clamp(max=1).float()
    with torch.no_grad():
        for gradient in gradients:
            gradient.mul_(coefficient.to(dtype=gradient.dtype))
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return {"applied": True, "grad_norm": float(norm),
            "scale_before": scale_before, "scale_after": float(scaler.get_scale())}
