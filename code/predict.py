"""Real-image inference and side-by-side ground-truth/prediction rendering."""
import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from boxes import decode_predictions
from dataset import VOC_CLASSES, image_tensor
from run_utils import load_detector, write_json


def draw_boxes(image, detections, threshold=0.25, title="Predictions"):
    image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for item in detections:
        if item.get("score", 1) < threshold:
            continue
        box = [item["box"][0] * width, item["box"][1] * height,
               item["box"][2] * width, item["box"][3] * height]
        draw.rectangle(box, outline="lime", width=2)
        text = VOC_CLASSES[item["class_id"]]
        if "score" in item:
            text += f" {item['score']:.2f}"
        if item.get("difficult"):
            text += " difficult"
        draw.text((box[0], max(14, box[1]) - 12), text, fill="yellow", stroke_width=1, stroke_fill="black")
    draw.rectangle((0, 0, min(width, 230), 16), fill="black")
    draw.text((3, 2), title, fill="white")
    return image


def save_dataset_examples(dataset, predictions, output, count=10, threshold=0.25):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for image_id in dataset.ids[:count]:
        with Image.open(dataset.root / "JPEGImages" / f"{image_id}.jpg") as source:
            image = source.convert("RGB")
        ann = dataset.annotations[image_id]
        ground = [{"box": b.tolist(), "class_id": int(c), "difficult": bool(d)}
                  for b, c, d in zip(ann["boxes"], ann["labels"], ann["difficult"])]
        left = draw_boxes(image, ground, title="Ground truth")
        right = draw_boxes(image, predictions[image_id], threshold, "Predictions")
        canvas = Image.new("RGB", (image.width * 2, image.height))
        canvas.paste(left, (0, 0)); canvas.paste(right, (image.width, 0))
        canvas.save(output / f"{image_id}.jpg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", default="prediction.jpg", type=Path)
    parser.add_argument("--threshold", default=0.25, type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    model, checkpoint = load_detector(args.checkpoint, args.device)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    model.eval()
    with torch.inference_mode():
        output = model(image_tensor(image).unsqueeze(0).to(args.device))
    detections = decode_predictions(output, args.threshold, checkpoint["config"].get("nms_threshold", 0.5))[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    draw_boxes(image, detections, args.threshold).save(args.output)
    write_json(args.output.with_suffix(".json"), detections)


if __name__ == "__main__":
    main()
