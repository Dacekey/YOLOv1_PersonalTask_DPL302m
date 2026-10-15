"""PASCAL VOC XML loader. Evaluation keeps every GT, including difficult objects."""
import argparse
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

VOC_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
               "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
               "pottedplant", "sheep", "sofa", "train", "tvmonitor")


def find_voc_root(path):
    path = Path(path)
    candidates = [path] + sorted(path.glob("**/VOC2007"))
    matches = [p for p in candidates if (p / "Annotations").is_dir() and
               (p / "JPEGImages").is_dir() and (p / "ImageSets/Main").is_dir()]
    matches = list(dict.fromkeys(p.resolve() for p in matches))
    if len(matches) != 1:
        raise ValueError(f"Expected one VOC2007 root under {path}; found {matches}. Set --data explicitly.")
    return matches[0]


def read_annotation(path):
    root = ET.parse(path).getroot()
    width, height = float(root.findtext("size/width")), float(root.findtext("size/height"))
    boxes, labels, difficult = [], [], []
    for obj in root.findall("object"):
        label = obj.findtext("name")
        if label not in VOC_CLASSES:
            raise ValueError(f"Unknown VOC class {label!r} in {path}")
        b = obj.find("bndbox")
        # VOC uses one-based inclusive endpoints; convert to continuous image edges.
        box = [(float(b.findtext("xmin")) - 1) / width,
               (float(b.findtext("ymin")) - 1) / height,
               float(b.findtext("xmax")) / width, float(b.findtext("ymax")) / height]
        box = np.clip(box, 0, 1).tolist()
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Degenerate annotation in {path}: {box}")
        boxes.append(box); labels.append(VOC_CLASSES.index(label))
        difficult.append(bool(int(obj.findtext("difficult", "0"))))
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
            "difficult": torch.tensor(difficult, dtype=torch.bool)}


def encode_target(annotation):
    grid = torch.zeros(7, 7, 25)
    ignore = torch.zeros(7, 7, dtype=torch.bool)
    boxes = annotation["boxes"]
    sizes = boxes[:, 2:] - boxes[:, :2]
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    cells = (centers * 7).long().clamp(0, 6)
    for cell, difficult in zip(cells, annotation["difficult"]):
        if difficult:
            ignore[cell[1], cell[0]] = True
    collisions = 0
    # Largest non-difficult object wins; stable index breaks equal-area ties.
    for index in torch.argsort(sizes.prod(-1), descending=True, stable=True).tolist():
        if annotation["difficult"][index]:
            continue
        col, row = cells[index].tolist()
        if grid[row, col, 20]:
            collisions += 1
            continue
        grid[row, col, annotation["labels"][index]] = 1
        grid[row, col, 20] = 1
        grid[row, col, 21:23] = centers[index] * 7 - cells[index]
        grid[row, col, 23:25] = sizes[index]
        ignore[row, col] = False
    return grid, ignore, collisions


def image_tensor(image):
    image = image.resize((448, 448), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0


class VOCDataset(Dataset):
    def __init__(self, root, split="train", augment=False, limit=None, ids=None):
        self.root = find_voc_root(root)
        split_path = self.root / "ImageSets/Main" / f"{split}.txt"
        if ids is None:
            if not split_path.is_file():
                raise FileNotFoundError(f"Official split is missing: {split_path}")
            ids = [line.split()[0] for line in split_path.read_text().splitlines() if line.strip()]
        self.ids = list(ids)[:limit] if limit else list(ids)
        if not self.ids or len(self.ids) != len(set(self.ids)):
            raise ValueError("Split is empty or contains duplicate IDs")
        self.split, self.augment = split, augment
        self.annotations = {}
        for image_id in self.ids:
            if not (self.root / "JPEGImages" / f"{image_id}.jpg").is_file():
                raise FileNotFoundError(f"Missing image {image_id}")
            self.annotations[image_id] = read_annotation(self.root / "Annotations" / f"{image_id}.xml")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        with Image.open(self.root / "JPEGImages" / f"{image_id}.jpg") as source:
            image = source.convert("RGB")
        annotation = {k: v.clone() for k, v in self.annotations[image_id].items()}
        if self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
                old = annotation["boxes"].clone()
                annotation["boxes"][:, 0] = 1 - old[:, 2]
                annotation["boxes"][:, 2] = 1 - old[:, 0]
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.8, 1.2))
        grid, ignore, collisions = encode_target(annotation)
        return image_tensor(image), {**annotation, "image_id": image_id,
                                    "grid": grid, "ignore": ignore, "collisions": collisions}


def collate(batch):
    images, annotations = zip(*batch)
    return torch.stack(images), list(annotations)


def inspect_dataset(path):
    root = find_voc_root(path)
    result = {"root": str(root), "splits": {}}
    for split in ("train", "val", "trainval", "test"):
        if not (root / "ImageSets/Main" / f"{split}.txt").exists():
            result["splits"][split] = {"available": False}
            continue
        dataset = VOCDataset(root, split)
        result["splits"][split] = {"available": True, "images": len(dataset),
            "objects": sum(len(a["labels"]) for a in dataset.annotations.values()),
            "difficult": sum(int(a["difficult"].sum()) for a in dataset.annotations.values()),
            "collisions": sum(encode_target(a)[2] for a in dataset.annotations.values())}
    train = set(VOCDataset(root, "train").ids)
    val = set(VOCDataset(root, "val").ids)
    if train & val:
        raise ValueError("Official train and val overlap")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = json.dumps(inspect_dataset(args.data), indent=2)
    print(report)
    if args.output:
        Path(args.output).write_text(report + "\n")
