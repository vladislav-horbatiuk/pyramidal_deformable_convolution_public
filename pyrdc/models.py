"""Small experiment models: a CIFAR-scale classifier and a compact U-Net.

Both build a feature pyramid and hand it to an exchangeable head created by
`make_head`, so every head type (plain conv, deformable conv, multi-scale
fusion, pyramidal variants) is compared inside an otherwise identical network.

Note: the notebook versions kept the 1x1 adapter convs in a plain Python list,
so their parameters were invisible to the optimizer, `.to(device)` and
`state_dict()`. Here they live in an nn.ModuleList.
"""
import torch
import torch.nn as nn
from torchvision.ops import Conv2dNormActivation

from .blocks import make_head


def _head_uses_pyramid(head_type):
    """Heads that read levels other than pyramid[-1] (and hence need adapters)."""
    return head_type.startswith("pyramidal") and "single_level" not in head_type


def _adapters(num, channels):
    return nn.ModuleList(
        nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True))
        for _ in range(num)
    )


class SmallClassifier(nn.Module):
    """4-block conv backbone (stride 2, 2, 2, 1) + exchangeable head + MLP classifier.

    The pyramid handed to the head is [x1, x2, x3, x4] (coarse-to-fine in depth,
    fine-to-coarse spatially: /2, /4, /8, /8), with x4 as the reference level.
    """

    def __init__(self, head_type, num_classes, channels=16, use_adapters=True,
                 **head_kwargs):
        super().__init__()
        strides = (2, 2, 2, 1)
        in_chs = (3, channels, channels, channels)
        self.blocks = nn.ModuleList(
            Conv2dNormActivation(
                in_channels=ic, out_channels=channels, kernel_size=3, stride=s, padding=1
            )
            for ic, s in zip(in_chs, strides)
        )
        self.head = make_head(head_type, channels, channels, num_levels=4, **head_kwargs)
        self.use_adapters = use_adapters and _head_uses_pyramid(head_type)
        self.adapters = _adapters(3, channels) if self.use_adapters else None
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(channels * 16, 164),
            nn.ReLU(inplace=True),
            nn.Linear(164, 148),
            nn.ReLU(inplace=True),
            nn.Linear(148, num_classes),
        )

    def forward(self, x):
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        pyramid = feats
        if self.use_adapters:
            pyramid = [a(f) for a, f in zip(self.adapters, feats[:-1])] + [feats[-1]]
        out = self.head(pyramid)
        return self.classifier(out)


class SmallUNet(nn.Module):
    """Compact 2-level U-Net whose final conv is an exchangeable head.

    The pyramid handed to the head is [p1, b, d2, d1] where d1 (full resolution)
    is the reference level.
    """

    def __init__(self, head_type, in_channels=3, out_channels=1, channels=16,
                 skip_connections=True, use_adapters=True, **head_kwargs):
        super().__init__()
        self.enc1 = self._conv_block(in_channels, channels)
        self.enc2 = self._conv_block(channels, channels)
        self.bottleneck = self._conv_block(channels, channels)
        dec_in = channels * 2 if skip_connections else channels
        self.dec2 = self._conv_block(dec_in, channels)
        self.dec1 = self._conv_block(dec_in, channels)
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.skip_connections = skip_connections

        self.head = make_head(head_type, channels, out_channels, num_levels=4, **head_kwargs)
        self.use_adapters = use_adapters and _head_uses_pyramid(head_type)
        self.adapters = _adapters(3, channels) if self.use_adapters else None

    @staticmethod
    def _conv_block(in_c, out_c):
        return Conv2dNormActivation(
            in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1, inplace=True
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        b = self.bottleneck(p2)

        d2 = self.upsample(b)
        if self.skip_connections:
            d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upsample(d2)
        if self.skip_connections:
            d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        pyramid = [p1, b, d2, d1]
        if self.use_adapters:
            pyramid = [a(f) for a, f in zip(self.adapters, pyramid[:-1])] + [d1]
        return self.head(pyramid)
