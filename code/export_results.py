"""Package a completed or interrupted run, without including data or repository files."""
import argparse
import csv
import zipfile
from pathlib import Path

from run_utils import file_sha256, write_json


def plot_history(output):
    history = Path(output) / "history.csv"
    if not history.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with history.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [int(r["epoch"]) for r in rows]
    axes[0].plot(epochs, [float(r["train_loss"]) for r in rows], label="train")
    axes[0].plot(epochs, [float(r["val_loss"]) for r in rows], label="val")
    axes[0].set_ylabel("YOLOv1 loss"); axes[0].legend()
    axes[1].plot(epochs, [float(r["map50"]) for r in rows])
    axes[1].set_ylabel("VOC2007 mAP@0.5")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(Path(output) / "curves.png"); plt.close(fig)


def export_results(output, destination):
    output, destination = Path(output).resolve(), Path(destination).resolve()
    if not output.is_dir():
        raise FileNotFoundError(output)
    if destination == output or output in destination.parents:
        raise ValueError("ZIP must be outside the run directory")
    plot_history(output)
    files = [p for p in sorted(output.rglob("*")) if p.is_file() and
             not p.name.endswith(".tmp") and p.name != "artifact_checksums.json"]
    write_json(output / "artifact_checksums.json",
               {str(p.relative_to(output)): file_sha256(p) for p in files})
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1,
                         allowZip64=True) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                archive.write(path, Path(output.name) / path.relative_to(output))
    temp.replace(destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--zip", required=True)
    args = parser.parse_args()
    print(export_results(args.output, args.zip))
