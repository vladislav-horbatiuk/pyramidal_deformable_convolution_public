"""Segmentation losses."""
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        """logits: (B, 1, H, W); targets: (B, H, W) in {0, 1}."""
        probs = torch.sigmoid(logits).squeeze(1)
        targets = targets.float()
        intersection = (probs * targets).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce = self.bce(logits.squeeze(1), targets.float())
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice
