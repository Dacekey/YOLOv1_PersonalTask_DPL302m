"""Strict reader for the supplied legacy Darknet 16-convolution checkpoint."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn

EXPECTED_SHA256 = "048f832fbaddec03256f2c1cf380e5694d6b580f24639515587daa358ff32e9f"
EXPECTED_BYTES = 47_657_232


def load_pretrained(backbone, path, verify_hash=True):
    path = Path(path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != EXPECTED_BYTES:
        raise ValueError(f"Expected {EXPECTED_BYTES} bytes for the 16-conv file; got {len(data)}")
    if verify_hash and digest != EXPECTED_SHA256:
        raise ValueError("Pretrained SHA-256 differs from the verified extraction.conv.weights")
    header = struct.unpack("<4i", data[:16])
    if header[:3] != (0, 1, 0):
        raise ValueError(f"Unsupported legacy Darknet header: {header}")
    values = np.frombuffer(data, dtype="<f4", offset=16)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite pretrained values")
    layers = [(name, layer) for name, layer in backbone.named_children()
              if isinstance(layer, nn.Conv2d)][:16]
    from yolov1 import ARCHITECTURE
    expected = []
    channels = 3
    for spec in ARCHITECTURE:
        if spec == "M":
            continue
        out_channels, kernel, _, _ = spec
        expected.append((out_channels, channels, kernel, kernel))
        channels = out_channels
    if [tuple(layer.weight.shape) for _, layer in layers] != expected[:16]:
        raise ValueError("Pretrained convolution shapes differ from conv01-conv16")
    if len(layers) != 16 or sum(p.numel() for _, m in layers for p in (m.bias, m.weight)) != len(values):
        raise ValueError("Backbone does not match the supplied 16-conv weights")
    offset = 0
    with torch.no_grad():
        for _, layer in layers:
            for parameter in (layer.bias, layer.weight):
                count = parameter.numel()
                tensor = torch.from_numpy(values[offset:offset + count].copy()).reshape(parameter.shape)
                parameter.copy_(tensor)
                offset += count
    assert offset == len(values)
    return {"path": str(path), "sha256": digest, "bytes": len(data),
            "header": list(header), "loaded_layers": [name for name, _ in layers],
            "loaded_parameters": offset,
            "provenance": "Binary layout verified; training dataset/history not independently verified."}


if __name__ == "__main__":
    from yolov1 import YOLOv1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    args = parser.parse_args()
    print(json.dumps(load_pretrained(YOLOv1._make_backbone(), args.weights), indent=2))
