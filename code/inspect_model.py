"""Print the layer trace and verify a real YOLOv1 forward pass.

Example: conda run -n DeepAI python inspect_model.py --device cpu
Use --meta to check architecture/shapes without allocating parameter memory.
"""
import argparse
import json
from pathlib import Path
import platform

import torch
from torch import nn

from yolov1 import YOLOv1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--meta", action="store_true", help="Shape check only; no real numerical computation")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--save-json", type=Path, help="Write layer trace and verification results")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.device == "cuda" and not args.meta and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    device = "meta" if args.meta else args.device
    with torch.device(device):
        model = YOLOv1().eval()
    convolutions = [layer for layer in model.modules() if isinstance(layer, nn.Conv2d)]
    pools = [layer for layer in model.modules() if isinstance(layer, nn.MaxPool2d)]
    fully_connected = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
    # Check architectural invariants, including the four added convolutions.
    assert len(convolutions) == 24 and len(pools) == 4 and len(fully_connected) == 2
    assert [i for i, layer in enumerate(convolutions, 1) if layer.stride == (2, 2)] == [1, 22]
    assert all(layer.kernel_size == (3, 3) and layer.out_channels == 1024
               for layer in convolutions[20:])
    assert fully_connected[0].in_features == 50176 and fully_connected[0].out_features == 4096
    assert fully_connected[1].in_features == 4096 and fully_connected[1].out_features == 1470
    assert not any(isinstance(layer, (nn.BatchNorm2d, nn.Sigmoid)) for layer in model.modules())
    assert isinstance(model.head[-1], nn.Linear)
    trace = []
    handles = []

    def record(name):
        def hook(layer, inputs, output):
            trace.append({
                "name": name, "type": type(layer).__name__,
                "kernel": list(layer.kernel_size) if isinstance(layer, nn.Conv2d) else layer.kernel_size if isinstance(layer, nn.MaxPool2d) else None,
                "stride": list(layer.stride) if isinstance(layer, nn.Conv2d) else layer.stride if isinstance(layer, nn.MaxPool2d) else None,
                "output_shape": list(output.shape),
            })
        return hook

    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.Linear)):
            handles.append(layer.register_forward_hook(record(name)))
    images = torch.zeros(1, 3, 448, 448, device=device)
    with torch.inference_mode():
        predictions = model(images)
    for handle in handles:
        handle.remove()
    assert tuple(predictions.shape) == (1, 7, 7, 30)
    finite = None if args.meta else bool(torch.isfinite(predictions).all().item())
    assert args.meta or finite
    # The fixed-size FC predictor must reject an incompatible input.
    try:
        model(torch.zeros(1, 3, 224, 224, device=device))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected incompatible input to be rejected")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "model": "YOLOv1", "device": device, "python": platform.python_version(),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "input_shape": list(images.shape), "output_shape": list(predictions.shape),
        "conv_count": len(convolutions), "pool_count": len(pools),
        "fc_count": len(fully_connected), "parameters": parameter_count,
        "parameter_memory_mib_float32": parameter_count * 4 / 1024**2,
        "numerical_forward_performed": not args.meta, "output_finite": finite,
        "architecture_checks_passed": True, "layers": trace,
    }
    for row in trace:
        print(f'{row["name"]:18} {row["type"]:10} stride={str(row["stride"]):6} -> {row["output_shape"]}')
    print(f"Parameters: {parameter_count:,} (~{result['parameter_memory_mib_float32']:.1f} MiB in float32)")
    print(f"Input: {list(images.shape)}; output: {list(predictions.shape)}; device: {device}")
    print(f"Architecture checks passed; numerical forward: {not args.meta}; finite output: {finite}")
    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
