"""YOLOv1 sum-squared error, divided by batch size (not grid cells)."""
import torch
from torch import nn
from boxes import box_iou, grid_boxes


class YOLOv1Loss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.lambda_coord, self.lambda_noobj = lambda_coord, lambda_noobj

    def forward(self, predictions, targets, ignore=None):
        predictions = predictions.float()
        if predictions.shape[1:] != (7, 7, 30) or targets.shape[1:] != (7, 7, 25):
            raise ValueError("Expected predictions [N,7,7,30], targets [N,7,7,25]")
        boxes = predictions[..., 20:].reshape(-1, 7, 7, 2, 5)
        target_boxes = targets[..., 21:25].unsqueeze(-2)
        obj = targets[..., 20].bool()
        with torch.no_grad():
            ious = box_iou(grid_boxes(boxes[..., :4], True), grid_boxes(target_boxes))
            responsible = torch.nn.functional.one_hot(ious.argmax(-1), 2).bool() & obj.unsqueeze(-1)
            confidence_target = ious.detach()
        noobj = ~responsible
        if ignore is not None:
            noobj = noobj & ~ignore.unsqueeze(-1)
        xy_error = (boxes[..., :2] - target_boxes[..., :2]).square().sum(-1)
        # Raw predicted sizes parameterize sqrt(w/h); decode by squaring in IoU/inference.
        wh_error = (boxes[..., 2:4] - target_boxes[..., 2:4].clamp_min(0).sqrt()).square().sum(-1)
        terms = {
            "xy": self.lambda_coord * xy_error[responsible].sum(),
            "wh": self.lambda_coord * wh_error[responsible].sum(),
            "object": (boxes[..., 4] - confidence_target).square()[responsible].sum(),
            "no_object": self.lambda_noobj * boxes[..., 4].square()[noobj].sum(),
            "class": (predictions[..., :20] - targets[..., :20]).square().sum(-1)[obj].sum(),
        }
        terms = {k: v / predictions.shape[0] for k, v in terms.items()}
        total = sum(terms.values())
        return total, {k: float(v.detach()) for k, v in terms.items()}
