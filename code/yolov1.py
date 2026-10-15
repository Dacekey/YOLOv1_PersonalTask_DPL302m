"""YOLOv1 in PyTorch: Figure 3 of Redmon et al., CVPR 2016.

24 convolutional layers, four max-pooling layers, and two fully connected
layers. No batch normalization, anchors, sigmoid, or adaptive pooling.
All parameters start randomly initialized; ImageNet pretraining and detector
training are separate tasks. This module does not load Darknet weights.
"""
from collections import OrderedDict

import torch
from torch import Tensor, nn


# (output channels, kernel size, stride, padding); "M" denotes max pooling.
# Convolution numbering follows the original paper, excluding pool/FC layers.
ARCHITECTURE: tuple[tuple[int, int, int, int] | str, ...] = (
    (64, 7, 2, 3),       # Conv 01: 448 -> 224
    "M",                # 224 -> 112
    (192, 3, 1, 1),      # Conv 02
    "M",                # 112 -> 56
    (128, 1, 1, 0),      # Conv 03
    (256, 3, 1, 1),      # Conv 04
    (256, 1, 1, 0),      # Conv 05
    (512, 3, 1, 1),      # Conv 06
    "M",                # 56 -> 28
    (256, 1, 1, 0),      # Conv 07
    (512, 3, 1, 1),      # Conv 08
    (256, 1, 1, 0),      # Conv 09
    (512, 3, 1, 1),      # Conv 10
    (256, 1, 1, 0),      # Conv 11
    (512, 3, 1, 1),      # Conv 12
    (256, 1, 1, 0),      # Conv 13
    (512, 3, 1, 1),      # Conv 14
    (512, 1, 1, 0),      # Conv 15
    (1024, 3, 1, 1),     # Conv 16
    "M",                # 28 -> 14
    (512, 1, 1, 0),      # Conv 17
    (1024, 3, 1, 1),     # Conv 18
    (512, 1, 1, 0),      # Conv 19
    (1024, 3, 1, 1),     # Conv 20: end of the pretrained convolutional portion
    (1024, 3, 1, 1),     # Conv 21: first added detection convolution
    (1024, 3, 2, 1),     # Conv 22: 14 -> 7
    (1024, 3, 1, 1),     # Conv 23
    (1024, 3, 1, 1),     # Conv 24
)


class YOLOv1(nn.Module):
    """Original full-size YOLOv1 detector, with configurable class count.

    Input:  floating-point RGB tensor [N, 3, 448, 448].
    Output: raw prediction tensor [N, 7, 7, C + 5*B], with no final activation.

    This implementation's per-cell convention is:
      [C class values, box_0(x,y,w,h,confidence), box_1(...), ...].
    Values require training and consistent target encoding/decoding; raw
    linear outputs are not necessarily valid probabilities or positive sizes.
    A loss or decoder must handle the square-root box-size convention chosen
    for training. Forward alone does not produce final image detections.
    """

    input_size = 448
    grid_size = 7

    def __init__(
        self,
        num_classes: int = 20,
        boxes_per_cell: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if num_classes < 1 or boxes_per_cell < 1:
            raise ValueError("num_classes and boxes_per_cell must be positive")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be in [0, 1]")
        self.num_classes = num_classes
        self.boxes_per_cell = boxes_per_cell
        self.values_per_cell = num_classes + 5 * boxes_per_cell
        self.backbone = self._make_backbone()
        self.head = nn.Sequential(OrderedDict([
            ("flatten", nn.Flatten(start_dim=1)),
            ("fc1", nn.Linear(1024 * self.grid_size * self.grid_size, 4096)),
            ("activation", nn.LeakyReLU(negative_slope=0.1, inplace=True)),
            ("dropout", nn.Dropout(p=dropout)),
            ("fc2", nn.Linear(4096, self.grid_size**2 * self.values_per_cell)),
        ]))

    @staticmethod
    def _make_backbone() -> nn.Sequential:
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        in_channels = 3
        conv_number = pool_number = 0
        for specification in ARCHITECTURE:
            if specification == "M":
                pool_number += 1
                layers[f"pool{pool_number}"] = nn.MaxPool2d(kernel_size=2, stride=2)
                continue
            out_channels, kernel, stride, padding = specification
            conv_number += 1
            layers[f"conv{conv_number:02d}"] = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel,
                stride=stride, padding=padding, bias=True,
            )
            layers[f"relu{conv_number:02d}"] = nn.LeakyReLU(0.1, inplace=True)
            in_channels = out_channels
        return nn.Sequential(layers)

    def forward(self, images: Tensor) -> Tensor:
        expected = (3, self.input_size, self.input_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(f"Expected input [N, 3, 448, 448], got {tuple(images.shape)}")
        features = self.backbone(images)
        predictions = self.head(features)
        return predictions.reshape(
            images.shape[0], self.grid_size, self.grid_size, self.values_per_cell,
        )


if __name__ == "__main__":
    # Import is kept lightweight. Constructing the full model needs ~1 GiB
    # just for float32 parameters because FC1 alone has ~205.5M weights.
    print("YOLOv1: 24 conv + 4 max pool + FC(4096) + FC(1470)")
    print("Use: python inspect_model.py --device cpu")
