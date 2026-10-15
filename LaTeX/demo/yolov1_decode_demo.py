"""Deterministic YOLOv1 decoding/NMS example, without a trained network.

Run from any directory: python demo/yolov1_decode_demo.py
The fixture uses [20 class scores, box 0, box 1] in each cell. Each box is
[cell-relative x, cell-relative y, image-normalized w, h, confidence].
Widths/heights are already decoded, not square-root raw network outputs.
This educational layout is not a Darknet binary-weight/output loader.
"""
from pathlib import Path
import json
import os
import platform
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yolo-report-matplotlib")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


def iou(a, b):
    """IoU of two normalized xyxy boxes; degenerate boxes have IoU zero."""
    a, b = np.asarray(a), np.asarray(b)
    extent = np.maximum(0.0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(extent))
    area_a = float(np.prod(np.maximum(0.0, a[2:] - a[:2])))
    area_b = float(np.prod(np.maximum(0.0, b[2:] - b[:2])))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def decode(prediction, score_threshold=0.25):
    """Convert the canonical (7,7,30) fixture into per-class candidates."""
    if prediction.shape != (7, 7, 30):
        raise ValueError("Expected the PASCAL VOC tensor shape (7, 7, 30)")
    candidates = []
    for row in range(7):
        for col in range(7):
            class_probabilities = prediction[row, col, :20]
            for predictor in range(2):
                start = 20 + 5 * predictor
                x, y, w, h, confidence = prediction[row, col, start:start + 5]
                if w <= 0 or h <= 0:
                    continue
                cx, cy = (col + x) / 7, (row + y) / 7
                box = np.clip([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 0, 1)
                for class_id, probability in enumerate(class_probabilities):
                    score = float(probability * confidence)
                    if score >= score_threshold:
                        candidates.append({
                            "id": f"r{row}c{col}b{predictor}:{VOC_CLASSES[class_id]}",
                            "class_id": class_id,
                            "label": VOC_CLASSES[class_id],
                            "score": score,
                            "box": box.tolist(),
                        })
    return candidates


def classwise_nms(candidates, iou_threshold=0.50):
    """Greedy NMS: overlapping boxes are suppressed only within a class."""
    remaining = sorted(candidates, key=lambda d: (-d["score"], d["id"]))
    kept, suppressed = [], []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        next_remaining = []
        for other in remaining:
            if other["class_id"] == best["class_id"] and iou(best["box"], other["box"]) > iou_threshold:
                suppressed.append(other)
            else:
                next_remaining.append(other)
        remaining = next_remaining
    return kept, suppressed


def fixture():
    prediction = np.zeros((7, 7, 30), dtype=np.float64)
    dog, person = VOC_CLASSES.index("dog"), VOC_CLASSES.index("person")
    prediction[3, 2, dog] = 0.9
    prediction[3, 2, 20:25] = [0.45, 0.45, 0.30, 0.35, 0.90]
    prediction[3, 2, 25:30] = [0.50, 0.50, 0.30, 0.35, 0.75]
    prediction[3, 4, person] = 0.8
    prediction[3, 4, 20:25] = [0.40, 0.30, 0.18, 0.60, 0.85]
    # An overlapping person/dog pair shows that NMS is class-specific.
    prediction[5, 5, dog] = 0.6
    prediction[5, 5, person] = 0.4
    prediction[5, 5, 20:25] = [0.25, 0.25, 0.18, 0.18, 0.80]
    prediction[5, 5, 25:30] = [0.27, 0.27, 0.18, 0.18, 0.20]
    return prediction


def draw_demo(candidates, kept):
    colors = {"dog": "#227C9D", "person": "#B65B32"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.45))
    for ax, detections, title in zip(axes, [candidates, kept], ["Score-filtered candidates", "After class-wise NMS"]):
        ax.set_facecolor("#F4F7FA")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(8) / 7)
        ax.set_yticks(np.arange(8) / 7)
        ax.grid(color="#CBD5E1", lw=0.45)
        ax.tick_params(labelbottom=False, labelleft=False, length=0)
        ax.set_title(f"{title}\n{len(detections)} class-specific predictions", fontsize=10, pad=10)
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, lw=1.4, edgecolor=colors[d["label"]]))
            # Separate nearly identical candidates and the cross-class pair.
            offset = 0.075 if d["id"].endswith(":person") and "r5c5" in d["id"] else 0
            if "b1" in d["id"]:
                offset = 0.09
            ax.text(x1+0.005, y1-0.014+offset, f'{d["label"]} {d["score"]:.3f}', fontsize=7.5,
                    color=colors[d["label"]], bbox={"facecolor":"white", "edgecolor":"none", "alpha":0.85, "pad":1})
        for spine in ax.spines.values():
            spine.set_color("#94A3B8")
    fig.text(0.5, 0.015, "Synthetic output tensor; no photograph, pretrained weights, or model inference.",
             ha="center", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    output = ROOT / "figures" / "demo_nms.pdf"
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    (ROOT / "figures").mkdir(exist_ok=True)
    prediction = fixture()
    candidates = decode(prediction)
    kept, suppressed = classwise_nms(candidates)
    duplicate_iou = iou(candidates[0]["box"], candidates[1]["box"])
    # Behavioral checks: scoring, geometry, duplicate removal, class separation.
    assert np.isclose(candidates[0]["score"], 0.81)
    assert np.isclose((candidates[0]["box"][0] + candidates[0]["box"][2]) / 2, 2.45 / 7)
    assert len(candidates) == 5 and len(kept) == 4 and len(suppressed) == 1
    assert suppressed[0]["id"] == "r3c2b1:dog"
    assert sum(d["id"].startswith("r5c5") for d in kept) == 2
    assert np.isclose(iou([0, 0, 1, 1], [0, 0, 1, 1]), 1)
    assert iou([0, 0, 0, 0], [0, 0, 1, 1]) == 0
    assert classwise_nms([]) == ([], [])
    result = {
        "description": "Synthetic YOLOv1 post-processing example, not trained-model evaluation",
        "tensor_shape": list(prediction.shape), "geometric_box_slots": 98,
        "score_threshold": 0.25, "nms_iou_threshold": 0.50,
        "candidates_before_nms": candidates, "detections_after_nms": kept,
        "suppressed": suppressed, "duplicate_iou": duplicate_iou,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "matplotlib": matplotlib.__version__, "platform": platform.platform()},
        "checks_passed": True,
    }
    (ROOT / "demo" / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    draw_demo(candidates, kept)
    print(f"Checks passed; candidates={len(candidates)}, kept={len(kept)}, suppressed={len(suppressed)}")
    print(f"Duplicate IoU={duplicate_iou:.6f}; surviving scores={[round(d['score'], 3) for d in kept]}")


if __name__ == "__main__":
    main()
