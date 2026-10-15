"""One shared box convention: xy offsets, raw sqrt sizes, normalized xyxy."""
import torch


def xywh_to_xyxy(boxes):
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def box_iou(a, b):
    """Broadcastable aligned IoU; pass a[:, None] and b[None] for pairwise."""
    intersection = (torch.minimum(a[..., 2:], b[..., 2:]) -
                    torch.maximum(a[..., :2], b[..., :2])).clamp_min(0).prod(-1)
    area_a = (a[..., 2:] - a[..., :2]).clamp_min(0).prod(-1)
    area_b = (b[..., 2:] - b[..., :2]).clamp_min(0).prod(-1)
    return intersection / (area_a + area_b - intersection).clamp_min(1e-12)


def grid_boxes(xywh, sqrt_sizes=False):
    """Decode [..., S, S, B, 4], without clipping training boxes."""
    size = xywh.shape[-3]
    row, col = torch.meshgrid(torch.arange(size, device=xywh.device),
                              torch.arange(size, device=xywh.device), indexing="ij")
    grid = torch.stack((col, row), -1).unsqueeze(-2)
    center = (xywh[..., :2] + grid) / size
    wh = xywh[..., 2:].square() if sqrt_sizes else xywh[..., 2:]
    return xywh_to_xyxy(torch.cat((center, wh), -1))


def nms(boxes, scores, threshold=0.5):
    order = scores.argsort(descending=True, stable=True)
    keep = []
    while order.numel():
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        rest = order[1:]
        order = rest[box_iou(boxes[current], boxes[rest]) <= threshold]
    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long, device=boxes.device)


def decode_predictions(predictions, score_threshold=0.001, nms_threshold=0.5, max_detections=300):
    """Linear class/confidence values; no sigmoid/softmax. Return CPU detections."""
    results = []
    for prediction in predictions.detach().float().cpu():
        classes = prediction.shape[-1] - 10
        raw = prediction[..., classes:].reshape(7, 7, 2, 5)
        boxes = grid_boxes(raw[..., :4], sqrt_sizes=True).reshape(-1, 4).clamp(0, 1)
        # Negative raw confidence/class values must not create positive scores together.
        scores = (prediction[..., :classes].clamp_min(0).unsqueeze(-2) *
                  raw[..., 4:].clamp_min(0)).reshape(-1, classes)
        valid = torch.isfinite(boxes).all(-1) & (boxes[:, 2:] > boxes[:, :2]).all(-1)
        output = []
        for class_id in range(classes):
            indices = torch.where(valid & torch.isfinite(scores[:, class_id]) &
                                  (scores[:, class_id] >= score_threshold))[0]
            kept = nms(boxes[indices], scores[indices, class_id], nms_threshold)
            for index in indices[kept].tolist():
                output.append({"class_id": class_id, "score": float(scores[index, class_id]),
                               "box": boxes[index].tolist()})
        results.append(sorted(output, key=lambda d: -d["score"])[:max_detections])
    return results
