"""Evaluate VOC validation, select three illustrative predictions, and export a small demo ZIP."""
import argparse
import csv
import json
import logging
import time
import zipfile
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from boxes import box_iou
from dataset import VOC_CLASSES, VOCDataset, collate
from evaluate import evaluate_model, save_metrics
from export_results import export_results
from run_utils import file_sha256, load_detector, runtime_info, seed_everything, write_json


def resolve_checkpoint(source, search_root, cache_dir):
    """Accept best.pt, a training ZIP, or a directory; extract only best.pt from a ZIP."""
    source = Path(source) if source else Path(search_root)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir():
        candidates = sorted(source.rglob("best.pt"))
        if len(candidates) > 1:
            raise ValueError(f"Multiple best.pt files; set CHECKPOINT_PATH explicitly: {candidates}")
        if candidates:
            return candidates[0], {"source": str(candidates[0]), "kind": "checkpoint"}
        archives = []
        for path in sorted(source.rglob("*.zip")):
            try:
                with zipfile.ZipFile(path) as archive:
                    if any(n.endswith("/checkpoints/best.pt") or n == "best.pt" for n in archive.namelist()):
                        archives.append(path)
            except zipfile.BadZipFile:
                continue
        if len(archives) != 1:
            raise ValueError(f"Expected one training ZIP or best.pt under {source}; found ZIPs: {archives}")
        source = archives[0]
    if source.suffix.lower() != ".zip":
        return source, {"source": str(source), "kind": "checkpoint"}
    with zipfile.ZipFile(source) as archive:
        members = [n for n in archive.namelist() if n.endswith("/checkpoints/best.pt") or n == "best.pt"]
        if len(members) != 1:
            raise ValueError(f"Expected one best.pt in {source}; found {members}")
        # Never extract the full archive or rely on its possibly misleading run name.
        cache = Path(cache_dir); cache.mkdir(parents=True, exist_ok=True)
        destination = cache / "best.pt"
        import shutil
        with archive.open(members[0]) as incoming, destination.with_suffix(".pt.tmp").open("wb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        destination.with_suffix(".pt.tmp").replace(destination)
    return destination, {"source": str(source), "kind": "training_zip", "member": members[0]}


def match_detections(detections, annotation, threshold=0.25, iou_threshold=0.5):
    """VOC matching for one image: same class, highest IoU, difficult ignored, duplicates FP."""
    matched = set()
    result = []
    for detection in sorted(detections, key=lambda d: -d["score"]):
        if detection["score"] < threshold:
            continue
        item = dict(detection)
        item.update({"class": VOC_CLASSES[item["class_id"]], "status": "FP", "iou": 0.0,
                     "gt_index": None})
        indices = torch.where(annotation["labels"] == item["class_id"])[0]
        if len(indices):
            overlaps = box_iou(torch.tensor(item["box"], dtype=torch.float32), annotation["boxes"][indices])
            overlap, index = overlaps.max(0)
            gt_index = int(indices[index])
            item.update(iou=float(overlap), gt_index=gt_index)
            if float(overlap) > iou_threshold:
                if bool(annotation["difficult"][gt_index]):
                    item["status"] = "ignored_difficult"
                elif gt_index not in matched:
                    item["status"] = "TP"
                    matched.add(gt_index)
        result.append(item)
    return result


def operating_metrics(predictions, annotations, threshold):
    per_image, all_matches = {}, {}
    tp = fp = gt = 0
    ious = []
    class_counts = [{"class": name, "tp": 0, "fp": 0, "gt": 0} for name in VOC_CLASSES]
    for image_id, annotation in annotations.items():
        matches = match_detections(predictions.get(image_id, []), annotation, threshold)
        all_matches[image_id] = matches
        positives = int((~annotation["difficult"]).sum())
        true = [m for m in matches if m["status"] == "TP"]
        false = sum(m["status"] == "FP" for m in matches)
        ious.extend(m["iou"] for m in true)
        tp += len(true); fp += false; gt += positives
        per_image[image_id] = {"image_id": image_id, "tp": len(true), "fp": false,
            "fn": positives - len(true), "gt_objects": positives, "detections": len(matches),
            "precision": len(true) / max(len(true) + false, 1),
            "recall": len(true) / max(positives, 1),
            "mean_tp_iou": sum(m["iou"] for m in true) / len(true) if true else None,
            "max_score": max((m["score"] for m in matches), default=0),
            "tp_classes": sorted({m["class_id"] for m in true})}
        for label, difficult in zip(annotation["labels"], annotation["difficult"]):
            if not difficult:
                class_counts[int(label)]["gt"] += 1
        for match in matches:
            if match["status"] == "TP":
                class_counts[match["class_id"]]["tp"] += 1
            elif match["status"] == "FP":
                class_counts[match["class_id"]]["fp"] += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(gt, 1)
    metrics = {"scope": "all evaluated validation images", "score_threshold": threshold,
        "iou_rule": "same class, IoU > 0.5; duplicate FP; difficult ignored",
        "images": len(annotations), "gt_objects": gt, "tp": tp, "fp": fp, "fn": gt - tp,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "mean_tp_iou": sum(ious) / len(ious) if ious else None,
        "images_with_predictions": sum(v["detections"] > 0 for v in per_image.values()),
        "images_with_correct_predictions": sum(v["tp"] > 0 for v in per_image.values()),
        "per_class": class_counts}
    return metrics, per_image, all_matches


def select_examples(per_image, count=3):
    candidates = [row for row in per_image.values() if row["detections"] > 0]
    # The selection explicitly uses GT to favor readable, correct illustrations.
    # This does not change predictions or metrics over the complete val split.
    candidates.sort(key=lambda row: (-int(row["tp"] > 0), -row["precision"],
        -(row["mean_tp_iou"] or 0), -row["max_score"], row["image_id"]))
    selected, used_classes = [], set()
    for row in candidates:
        if set(row["tp_classes"]) - used_classes:
            selected.append(row); used_classes.update(row["tp_classes"])
        if len(selected) == count:
            return selected
    selected_ids = {r["image_id"] for r in selected}
    for row in candidates:
        if row["image_id"] not in selected_ids:
            selected.append(row); selected_ids.add(row["image_id"])
        if len(selected) == count:
            break
    return selected


def csv_rows(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_panel(image, detections, title, show_iou=False):
    panel = Image.new("RGB", (image.width, image.height + 32), "#14202c")
    panel.paste(image, (0, 32))
    draw = ImageDraw.Draw(panel)
    label_font = font(max(12, min(18, image.width // 30)))
    draw.text((8, 6), title, fill="white", font=font(16))
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        box = [x1 * image.width, y1 * image.height + 32,
               x2 * image.width, y2 * image.height + 32]
        status = detection.get("status")
        color = "#34d399" if status in (None, "TP") else "#fb923c" if status == "FP" else "#c084fc"
        draw.rectangle(box, outline=color, width=3)
        label = VOC_CLASSES[detection["class_id"]]
        if "score" in detection:
            label += f" {detection['score']:.2f}"
        if show_iou:
            label += f" {status} IoU={detection['iou']:.2f}"
        elif detection.get("difficult"):
            label += " difficult"
        text_box = draw.textbbox((0, 0), label, font=label_font)
        width = text_box[2] - text_box[0] + 6
        height = text_box[3] - text_box[1] + 6
        left = max(0, min(int(box[0]), panel.width - width))
        top = max(32, int(box[1]) - height)
        draw.rectangle((left, top, left + width, top + height), fill="#14202c")
        draw.text((left + 3, top + 1), label, fill=color, font=label_font)
    return panel


def save_illustrations(dataset, selected, matches, output, threshold, max_boxes=10):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    comparisons, detection_rows = [], []
    for row in selected:
        image_id = row["image_id"]
        with Image.open(dataset.root / "JPEGImages" / f"{image_id}.jpg") as source:
            image = source.convert("RGB")
        annotation = dataset.annotations[image_id]
        ground = [{"box": box.tolist(), "class_id": int(label), "difficult": bool(difficult)}
                  for box, label, difficult in zip(annotation["boxes"], annotation["labels"], annotation["difficult"])]
        visible = matches[image_id][:max_boxes]
        panels = [render_panel(image, [], f"VOC2007 val: {image_id}"),
                  render_panel(image, ground, "Ground truth"),
                  render_panel(image, visible, f"Prediction: score >= {threshold:g}", show_iou=True)]
        composite = Image.new("RGB", (image.width * 3, image.height + 94), "#14202c")
        for index, panel in enumerate(panels):
            composite.paste(panel, (image.width * index, 0))
        draw = ImageDraw.Draw(composite)
        mean_iou = f"{row['mean_tp_iou']:.3f}" if row["mean_tp_iou"] is not None else "N/A"
        footer = (f"Illustrative selection | TP={row['tp']} FP={row['fp']} FN={row['fn']} | "
                  f"precision={row['precision']:.3f} recall={row['recall']:.3f} mean TP IoU={mean_iou}\n"
                  f"Green: TP; orange: FP; purple: difficult ignored. Showing {len(visible)}/{len(matches[image_id])} predictions.")
        draw.multiline_text((8, image.height + 38), footer, fill="white", font=font(13), spacing=5)
        composite.save(output / f"{image_id}_comparison.jpg", quality=95)
        panels[2].save(output / f"{image_id}_predictions.png")
        image.save(output / f"{image_id}_input.jpg", quality=95)
        comparisons.append(composite)
        for detection in matches[image_id]:
            detection_rows.append({"image_id": image_id, "class": detection["class"],
                "score": detection["score"], "status": detection["status"], "iou": detection["iou"],
                **dict(zip(("xmin", "ymin", "xmax", "ymax"), detection["box"]))})
    if comparisons:
        resized = []
        for image in comparisons:
            width = min(image.width, 1500)
            resized.append(image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS))
        sheet = Image.new("RGB", (max(im.width for im in resized), sum(im.height for im in resized)), "#14202c")
        top = 0
        for image in resized:
            sheet.paste(image, (0, top)); top += image.height
        sheet.save(output / "contact_sheet.jpg", quality=95)
    csv_rows(output / "selected_detections.csv", detection_rows)


def plot_ap(metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = metrics["per_class"]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([r["class"] for r in rows][::-1], [100 * (r["ap"] or 0) for r in rows][::-1], color="#227c9d")
    ax.set_xlabel("VOC2007 AP@0.5 (%)")
    scope = "Validation subset" if metrics.get("scope") == "limited validation subset" else "Full validation split"
    ax.set_title(f"{scope} — mAP@0.5 = {100 * metrics['map50']:.2f}%")
    ax.grid(axis="x", alpha=0.25); fig.tight_layout()
    fig.savefig(Path(output) / "per_class_ap.png", dpi=180)
    fig.savefig(Path(output) / "per_class_ap.pdf")
    plt.close(fig)


def run_demo(data, checkpoint, output, zip_path, device="cuda", batch_size=1, count=3,
             demo_threshold=0.25, limit=None, checkpoint_source=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}. Choose a new RUN_NAME.")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(42); torch.set_num_threads(4)
    logger = logging.getLogger("inference_demo"); logger.setLevel(logging.INFO); logger.propagate = False
    for old in list(logger.handlers):
        old.close(); logger.removeHandler(old)
    for handler in (logging.StreamHandler(), logging.FileHandler(output / "inference.log")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s")); logger.addHandler(handler)
    manifest = {**runtime_info(), "status": "running", "checkpoint_source": checkpoint_source or str(checkpoint),
                "dataset_source": "stpeteishii/pascal-voc-2007-dataset", "device": device,
                "purpose": "Educational demo of a self-defined YOLOv1 trained for 50 epochs; not a paper benchmark reproduction"}
    write_json(output / "run_manifest.json", manifest)
    started = time.perf_counter()
    try:
        model, saved = load_detector(checkpoint, device)
        if saved["config"].get("overfit"):
            raise ValueError("This is an overfit checkpoint. Attach best.pt/ZIP from the full-dataset 50-epoch training run.")
        config = dict(saved["config"])
        eval_threshold = config.get("eval_score_threshold", 0.001)
        if not eval_threshold <= demo_threshold <= 1 or count < 1 or batch_size < 1:
            raise ValueError("Require eval_score_threshold <= demo_threshold <= 1; positive count/batch_size")
        dataset = VOCDataset(data, "val", limit=limit)
        train_ids = set(saved.get("split_ids", {}).get("train", []))
        if train_ids & set(dataset.ids):
            raise ValueError("Validation overlaps the checkpoint's recorded training IDs")
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate,
                            num_workers=0, pin_memory=device.startswith("cuda"))
        manifest.update({"checkpoint_sha256": file_sha256(checkpoint),
            "checkpoint_epoch": saved["epoch"] + 1, "training_epochs_configured": config["epochs"],
            "training_config": config, "dataset_root": str(dataset.root), "split": "val",
            "validation_ids": dataset.ids, "validation_images": len(dataset),
            "evaluation_scope": "full validation split" if limit is None else "limited validation subset",
            "batch_size": batch_size, "demo_threshold": demo_threshold,
            "selection_policy": "Prefer images with same-class TP at IoU > 0.5, higher precision/IoU, then class diversity; fallback to images with any bbox. GT used only after inference for selection/evaluation.",
            "selected_examples_are_representative": False})
        write_json(output / "run_manifest.json", manifest)
        logger.info("Evaluating %d val images on %s; checkpoint epoch=%d", len(dataset), device, saved["epoch"] + 1)
        evaluation_started = time.perf_counter()
        def progress_batches():
            processed = 0
            for images, targets in loader:
                yield images, targets
                processed += len(targets)
                if processed % 100 < len(targets) or processed == len(dataset):
                    logger.info("Inference/decoding progress: %d/%d images", processed, len(dataset))
            logger.info("Forward/decoding finished; computing validation AP")
        metrics, predictions = evaluate_model(model, progress_batches(), device, config)
        evaluation_seconds = time.perf_counter() - evaluation_started
        metrics.update({"split": "val", "checkpoint_epoch": saved["epoch"] + 1,
                        "scope": manifest["evaluation_scope"]})
        save_metrics(metrics, predictions, output)
        operating, per_image, matches = operating_metrics(predictions, dataset.annotations, demo_threshold)
        selected = select_examples(per_image, count)
        write_json(output / "operating_point_metrics.json", operating)
        write_json(output / "selected_examples.json", {"selection_policy": manifest["selection_policy"],
            "requested_count": count, "actual_count": len(selected), "examples": selected})
        csv_rows(output / "per_image_metrics.csv", list(per_image.values()))
        csv_rows(output / "selected_examples.csv", selected)
        sweep = []
        for threshold in sorted({0.05, 0.1, demo_threshold, 0.5}):
            if threshold < eval_threshold:
                continue
            point = operating if threshold == demo_threshold else operating_metrics(predictions, dataset.annotations, threshold)[0]
            sweep.append({key: point[key] for key in ("score_threshold", "tp", "fp", "fn", "precision", "recall", "f1", "images_with_predictions")})
        csv_rows(output / "threshold_metrics.csv", sweep)
        save_illustrations(dataset, selected, matches, output / "examples", demo_threshold)
        plot_ap(metrics, output)
        explanation = (
            "YOLOv1 EDUCATIONAL INFERENCE DEMO\n"
            f"Checkpoint epoch: {saved['epoch'] + 1}; validation images: {len(dataset)}\n"
            f"Evaluation scope: {manifest['evaluation_scope']}\n"
            f"VOC2007 11-point mAP@0.5: {metrics['map50'] * 100:.4f}%\n"
            f"At score >= {demo_threshold:g}: precision={operating['precision']:.4f}, recall={operating['recall']:.4f}, F1={operating['f1']:.4f}\n"
            f"TP={operating['tp']}, FP={operating['fp']}, FN={operating['fn']}\n"
            f"Selected image IDs: {', '.join(row['image_id'] for row in selected)}\n"
            "The three illustrations are intentionally selected using GT after inference; they are not an unbiased performance sample.\n"
            "All-split metrics above are computed before selection. Per-example metrics are separate.\n"
            "Inference uses RGB [0,1], bilinear resize 448x448, raw square-root size decoding and class-wise NMS.\n"
            "Score is class value times confidence, not a calibrated probability. No extra ImageNet normalization is applied.\n"
            "This self-defined model and its 50-epoch recipe are an implementation demo, not a reproduction of the original paper baseline.\n"
            "Evaluation wall time includes input loading, model, decoding and AP calculation; it is not a model-only latency/FPS benchmark.\n")
        (output / "summary.txt").write_text(explanation)
        import shutil
        source_dir = output / "source"; source_dir.mkdir()
        for name in ("inference_demo.py", "dataset.py", "boxes.py", "evaluate.py", "loss.py", "yolov1.py", "predict.py", "run_utils.py", "export_results.py"):
            shutil.copyfile(Path(__file__).parent / name, source_dir / name)
        manifest.update(status="completed", selected_image_ids=[r["image_id"] for r in selected],
            selected_count=len(selected), map50=metrics["map50"],
            evaluation_wall_seconds=evaluation_seconds, session_seconds=time.perf_counter() - started)
        write_json(output / "run_manifest.json", manifest)
        logger.info("Done: mAP50=%.4f%%; selected=%s", 100 * metrics["map50"], manifest["selected_image_ids"])
        print(explanation)
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "run_manifest.json", manifest)
        logger.exception("Inference failed")
        raise
    finally:
        for handler in list(logger.handlers):
            handler.flush(); handler.close(); logger.removeHandler(handler)
        export_results(output, zip_path)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", default="", help="best.pt, training ZIP, or directory")
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--cache", default="/kaggle/temp/yolov1_inference_weights")
    parser.add_argument("--output", required=True)
    parser.add_argument("--zip", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--demo-threshold", type=float, default=.25)
    parser.add_argument("--limit", type=int, help="Diagnostics only; omit for full validation metrics")
    args = parser.parse_args()
    checkpoint, origin = resolve_checkpoint(args.checkpoint, args.input_root, args.cache)
    run_demo(args.data, checkpoint, args.output, args.zip, args.device, args.batch_size,
             args.count, args.demo_threshold, args.limit, origin)


if __name__ == "__main__":
    main()
