"""Evaluation metrics.

Segmentation metrics are accumulated over the whole dataset (sum of
intersections / unions) rather than averaged per batch, so results do not
depend on batch size and partial final batches are handled correctly.
"""
import torch


@torch.no_grad()
def segmentation_scores(logits, targets, threshold=0.5):
    """Returns per-batch (intersection, pred_sum, target_sum) sums for binary
    segmentation. logits: (B, 1, H, W); targets: (B, H, W)."""
    preds = (torch.sigmoid(logits).squeeze(1) > threshold).float()
    targets = targets.float()
    intersection = (preds * targets).sum()
    return intersection.item(), preds.sum().item(), targets.sum().item()


class SegmentationMeter:
    """Accumulates dataset-level Dice and IoU."""

    def __init__(self, eps=1e-6):
        self.eps = eps
        self.intersection = 0.0
        self.pred_sum = 0.0
        self.target_sum = 0.0

    def update(self, logits, targets):
        i, p, t = segmentation_scores(logits, targets)
        self.intersection += i
        self.pred_sum += p
        self.target_sum += t

    @property
    def dice(self):
        return (2 * self.intersection + self.eps) / (self.pred_sum + self.target_sum + self.eps)

    @property
    def iou(self):
        union = self.pred_sum + self.target_sum - self.intersection
        return (self.intersection + self.eps) / (union + self.eps)


class ConfusionMeter:
    """Accumulates a KxK confusion matrix for multi-class segmentation and
    derives mean IoU / pixel accuracy. Pixels labeled `ignore_index` are
    excluded (Cityscapes convention: 255)."""

    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, logits, targets):
        """logits: (B, K, H, W); targets: (B, H, W) long."""
        preds = logits.argmax(dim=1)
        valid = targets != self.ignore_index
        t, p = targets[valid].long(), preds[valid].long()
        idx = t * self.num_classes + p
        self.mat += torch.bincount(
            idx, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes).cpu()

    @property
    def miou(self):
        m = self.mat.float()
        inter = m.diag()
        union = m.sum(0) + m.sum(1) - inter
        present = union > 0
        if not present.any():
            return 0.0
        return (inter[present] / union[present]).mean().item()

    @property
    def pixel_acc(self):
        total = self.mat.sum().item()
        return self.mat.diag().sum().item() / max(total, 1)

    @property
    def per_class_iou(self):
        """IoU per class, NaN for classes absent from both GT and prediction.
        Lets a table separate frequent (road/building) from rare (pole/rider)
        classes, which mIoU averages together and pixel accuracy hides."""
        m = self.mat.float()
        inter = m.diag()
        union = m.sum(0) + m.sum(1) - inter
        iou = inter / union
        iou[union == 0] = float("nan")
        return iou

    @property
    def class_pixel_freq(self):
        """Fraction of labelled pixels belonging to each GT class."""
        gt = self.mat.sum(1).float()
        return gt / max(gt.sum().item(), 1.0)


@torch.no_grad()
def per_image_miou(logits, targets, num_classes, ignore_index=255):
    """Per-sample mean IoU over the classes present in that image's GT or
    prediction, shape (B,). The multi-class analogue of `per_image_dice`:
    gives paired samples for a Wilcoxon test between two heads."""
    preds = logits.argmax(dim=1)
    out = []
    for p, t in zip(preds, targets):
        valid = t != ignore_index
        p, t = p[valid].long(), t[valid].long()
        if p.numel() == 0:
            out.append(float("nan"))
            continue
        mat = torch.bincount(t * num_classes + p,
                             minlength=num_classes ** 2).reshape(num_classes, num_classes).float()
        inter = mat.diag()
        union = mat.sum(0) + mat.sum(1) - inter
        present = union > 0
        out.append((inter[present] / union[present]).mean().item())
    return torch.tensor(out)


@torch.no_grad()
def per_image_dice(logits, targets, threshold=0.5, eps=1e-6):
    """Per-sample Dice, shape (B,). Used for paired significance tests
    between heads trained with the same seed on the same test set."""
    preds = (torch.sigmoid(logits).squeeze(1) > threshold).float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1, 2))
    denom = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
    return (2 * intersection + eps) / (denom + eps)


@torch.no_grad()
def boundary_f1(logits, targets, tol=2, threshold=0.5):
    """Boundary F1 score: precision/recall of predicted boundary pixels within
    `tol` pixels of ground-truth boundary pixels. Sensitive to exactly the
    fine-vs-coarse trade-off that cross-scale sampling is supposed to improve.

    logits: (B, 1, H, W); targets: (B, H, W). Returns mean F1 over the batch.
    """
    import torch.nn.functional as F

    preds = (torch.sigmoid(logits).squeeze(1) > threshold).float()
    targets = targets.float()

    def boundary(mask):
        m = mask.unsqueeze(1)
        eroded = -F.max_pool2d(-m, 3, stride=1, padding=1)
        return (m - eroded).clamp(min=0).squeeze(1)

    def dilate(mask, r):
        return F.max_pool2d(
            mask.unsqueeze(1), 2 * r + 1, stride=1, padding=r
        ).squeeze(1)

    pb, tb = boundary(preds), boundary(targets)
    pb_d, tb_d = dilate(pb, tol), dilate(tb, tol)
    eps = 1e-6
    precision = (pb * tb_d).sum(dim=(1, 2)) / (pb.sum(dim=(1, 2)) + eps)
    recall = (tb * pb_d).sum(dim=(1, 2)) / (tb.sum(dim=(1, 2)) + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1.mean().item()
