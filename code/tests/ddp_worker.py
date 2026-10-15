"""Small-network worker for real two-process CPU/Gloo integration tests."""
import argparse
import json
import signal
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
import distributed_utils as distributed
import train


class SmallDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(3, 1470)

    def forward(self, images):
        return self.fc(self.dropout(images.mean((2, 3)))).reshape(-1, 7, 7, 30)


def load(path, device):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    model = SmallDetector().to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--interrupt", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    device = distributed.initialize("cpu")
    train.build_model = lambda config, device, pretrained_path: (SmallDetector().to(device), None)
    train.load_detector = load
    original_eval = train.evaluate_model
    calls = [0]
    def evaluate(*values, **kwargs):
        assert distributed.rank() == 0, "Validation must run on rank zero only"
        calls[0] += 1
        result = original_eval(*values, **kwargs)
        if args.interrupt and calls[0] == 1:
            signal.raise_signal(signal.SIGTERM)
        return result
    train.evaluate_model = evaluate
    config = {"epochs": 2, "batch_size": 1, "accumulation": 2, "learning_rate": .001,
              "seed": 42, "num_workers": args.workers, "cpu_threads": 1, "augment": True,
              "amp": False, "require_pretrained": False, "example_count": 1,
              "max_detections": 10, "export_reserve_seconds": 0, "log_every": 50,
              "expected_world_size": 2}
    try:
        train.train_run(config, args.data, args.output, resume=args.resume, device=device,
                        zip_path=str(args.output) + ".zip")
        # Both workers must reach this point with their local model/RNG lifecycle intact.
        Path(args.output, f"worker_{distributed.rank()}_finished.json").write_text(json.dumps({
            "validation_calls": calls[0], "rank": distributed.rank()}))
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
