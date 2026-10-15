"""VOC 2007 11-point AP at IoU 0.5, including difficult and duplicate handling."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from boxes import box_iou, decode_predictions
from dataset import VOC_CLASSES, VOCDataset, collate
from loss import YOLOv1Loss
from run_utils import load_detector, write_json


def voc_metrics(predictions, annotations, iou_threshold=0.5):
    rows = []
    for class_id, name in enumerate(VOC_CLASSES):
        records = {}
        positive = 0
        for image_id, annotation in annotations.items():
            mask = annotation["labels"] == class_id
            difficult = annotation["difficult"][mask]
            records[image_id] = {"boxes": annotation["boxes"][mask], "difficult": difficult,
                                 "matched": torch.zeros(int(mask.sum()), dtype=torch.bool)}
            positive += int((~difficult).sum())
        detections = [(image_id, d) for image_id, values in predictions.items()
                      for d in values if d["class_id"] == class_id]
        detections.sort(key=lambda pair: -pair[1]["score"])
        tp, fp = [], []
        for image_id, detection in detections:
            if image_id not in records:
                raise ValueError(f"Prediction has unknown image ID {image_id}")
            record = records[image_id]
            if len(record["boxes"]) == 0:
                tp.append(0); fp.append(1); continue
            ious = box_iou(torch.tensor(detection["box"]), record["boxes"])
            overlap, index = ious.max(0)
            # VOC devkit uses overlap > threshold (strictly greater).
            if float(overlap) > iou_threshold:
                if record["difficult"][index]:
                    continue
                duplicate = bool(record["matched"][index])
                tp.append(int(not duplicate)); fp.append(int(duplicate))
                record["matched"][index] = True
            else:
                tp.append(0); fp.append(1)
        tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
        recall = tp_cum / max(positive, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
        ap = float(np.mean([np.max(precision[recall >= t], initial=0)
                           for t in np.linspace(0, 1, 11)])) if positive else None
        rows.append({"class": name, "class_id": class_id, "gt_objects": positive,
                     "ap": ap, "tp": int(sum(tp)), "fp": int(sum(fp))})
    valid = [r["ap"] for r in rows if r["ap"] is not None]
    return {"protocol": "VOC2007 11-point AP, IoU > 0.5, difficult ignored",
            "map50": float(np.mean(valid)) if valid else 0.0,
            "classes_with_gt": len(valid), "classes_total": 20,
            "images": len(annotations), "per_class": rows}


@torch.inference_mode()
def evaluate_model(model, loader, device, config):
    model.eval()
    predictions, annotations, losses = {}, {}, []
    criterion = YOLOv1Loss(config.get("lambda_coord", 5), config.get("lambda_noobj", 0.5))
    for images, targets in loader:
        output = model(images.to(device, non_blocking=True))
        if not torch.isfinite(output).all():
            raise FloatingPointError("Non-finite validation predictions")
        grids = torch.stack([t["grid"] for t in targets]).to(device)
        ignored = torch.stack([t["ignore"] for t in targets]).to(device)
        value, _ = criterion(output, grids, ignored)
        losses.append((float(value), len(targets)))
        decoded = decode_predictions(output, config.get("eval_score_threshold", 0.001),
            config.get("nms_threshold", 0.5), config.get("max_detections", 300))
        for target, detection in zip(targets, decoded):
            image_id = target["image_id"]
            predictions[image_id] = detection
            annotations[image_id] = {k: target[k] for k in ("boxes", "labels", "difficult")}
    metrics = voc_metrics(predictions, annotations)
    metrics["loss"] = sum(v * n for v, n in losses) / max(sum(n for _, n in losses), 1)
    metrics["postprocess"] = {k: config.get(k, default) for k, default in
        (("eval_score_threshold", 0.001), ("nms_threshold", 0.5), ("max_detections", 300))}
    return metrics, predictions


def save_metrics(metrics, predictions, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", metrics)
    write_json(output / "predictions.json", predictions)
    with (output / "per_class_ap.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics["per_class"][0]))
        writer.writeheader(); writer.writerows(metrics["per_class"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output", default="results/evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    model, checkpoint = load_detector(args.checkpoint, args.device)
    dataset = VOCDataset(args.data, args.split, limit=args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)
    metrics, predictions = evaluate_model(model, loader, args.device, checkpoint["config"])
    metrics.update({"split": args.split, "checkpoint_epoch": checkpoint["epoch"]})
    save_metrics(metrics, predictions, args.output)
    print(f"{args.split}: mAP50={metrics['map50']:.6f}, images={len(dataset)}")


if __name__ == "__main__":
    main()
