"""Training / evaluation loops shared by the experiment scripts."""
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .losses import BCEDiceLoss
from .metrics import ConfusionMeter, SegmentationMeter, boundary_f1


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate_classification(model, loader, device, criterion=None):
    """Returns (accuracy, mean loss). The loss is what makes the train/val gap
    comparable across heads; pass the same criterion
    the training loop uses or the two curves are not on the same scale."""
    model.eval()
    correct = total = 0
    total_loss, n_batches = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        if criterion is not None:
            total_loss += criterion(logits, labels).item()
            n_batches += 1
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    return correct / total, (total_loss / n_batches if n_batches else float("nan"))


@torch.no_grad()
def evaluate_segmentation(model, loader, criterion, device):
    model.eval()
    meter = SegmentationMeter()
    total_loss = 0.0
    total_bf1 = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, masks).item()
        total_bf1 += boundary_f1(logits, masks)
        meter.update(logits, masks)
    n = len(loader)
    return {
        "loss": total_loss / n,
        "dice": meter.dice,
        "iou": meter.iou,
        "boundary_f1": total_bf1 / n,
    }


def train_classification(model, train_loader, test_loader, device, writer=None,
                         epochs=100, lr=1e-3, weight_decay=1e-5, eval_every=3,
                         log_every=100):
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    step = 0
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            if writer is not None and step % log_every == 0:
                writer.add_scalar("Loss/train", loss.item(), step)
            step += 1
        scheduler.step()
        if writer is not None:
            writer.add_scalar("epoch_loss/train", epoch_loss / n_batches, epoch)
        msg = f"epoch {epoch + 1}/{epochs} train_loss={epoch_loss / n_batches:.4f}"
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            acc, test_loss = evaluate_classification(
                model, test_loader, device, criterion)
            best_acc = max(best_acc, acc)
            if writer is not None:
                writer.add_scalar("Acc/test", acc, epoch)
                writer.add_scalar("Loss/test", test_loss, epoch)
            msg += f" test_acc={acc:.4f} test_loss={test_loss:.4f}"
        print(msg, flush=True)
    return {"best_acc": best_acc, "final_acc": acc, "final_test_loss": test_loss}


@torch.no_grad()
def evaluate_segmentation_multiclass(model, loader, criterion, device, num_classes):
    model.eval()
    meter = ConfusionMeter(num_classes)
    total_loss, n = 0.0, 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, masks).item()
        meter.update(logits, masks)
        n += 1
    return {"loss": total_loss / n, "miou": meter.miou, "pixel_acc": meter.pixel_acc}


def train_segmentation_multiclass(model, train_loader, test_loader, device,
                                  num_classes, writer=None, epochs=60, lr=1e-3,
                                  weight_decay=1e-5, eval_every=2, log_every=20):
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    step = 0
    best = {"miou": 0.0, "pixel_acc": 0.0}
    metrics = {}
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            if writer is not None and step % log_every == 0:
                writer.add_scalar("Loss/train", loss.item(), step)
            step += 1
        scheduler.step()
        msg = f"epoch {epoch}/{epochs} train_loss={epoch_loss / n_batches:.4f}"
        if epoch % eval_every == 0 or epoch == epochs:
            metrics = evaluate_segmentation_multiclass(
                model, test_loader, criterion, device, num_classes
            )
            for k in best:
                best[k] = max(best[k], metrics[k])
            if writer is not None:
                writer.add_scalar("Loss/test", metrics["loss"], epoch)
                writer.add_scalar("mIoU/test", metrics["miou"], epoch)
                writer.add_scalar("PixelAcc/test", metrics["pixel_acc"], epoch)
            msg += (f" val_loss={metrics['loss']:.4f} miou={metrics['miou']:.4f}"
                    f" pixel_acc={metrics['pixel_acc']:.4f}")
        print(msg, flush=True)
    return {"best": best, "final": metrics}


def train_segmentation(model, train_loader, test_loader, device, writer=None,
                       epochs=60, lr=1e-3, weight_decay=1e-5, eval_every=2,
                       log_every=20, offset_lr_mult=1.0, offset_no_decay=False):
    model.to(device)
    criterion = BCEDiceLoss()
    if offset_lr_mult != 1.0 or offset_no_decay:
        # DCN practice: train the offset branch slower (and optionally without
        # decay) than the rest of the network. Default path below is untouched.
        offset, rest = [], []
        for name, p in model.named_parameters():
            (offset if "offset_conv" in name else rest).append(p)
        optimizer = optim.Adam(
            [{"params": rest},
             {"params": offset, "lr": lr * offset_lr_mult,
              "weight_decay": 0.0 if offset_no_decay else weight_decay}],
            lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    step = 0
    best = {"dice": 0.0, "iou": 0.0, "boundary_f1": 0.0}
    metrics = {}
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        t0 = time.perf_counter()
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            if writer is not None and step % log_every == 0:
                writer.add_scalar("Loss/train", loss.item(), step)
            step += 1
        scheduler.step()
        if writer is not None:
            writer.add_scalar("epoch_loss/train", epoch_loss / n_batches, epoch)
            writer.add_scalar("epoch_time/train", time.perf_counter() - t0, epoch)
        msg = f"epoch {epoch}/{epochs} train_loss={epoch_loss / n_batches:.4f}"
        if epoch % eval_every == 0 or epoch == epochs:
            metrics = evaluate_segmentation(model, test_loader, criterion, device)
            for k in best:
                best[k] = max(best[k], metrics[k])
            if writer is not None:
                writer.add_scalar("Loss/test", metrics["loss"], epoch)
                writer.add_scalar("Dice/test", metrics["dice"], epoch)
                writer.add_scalar("IoU/test", metrics["iou"], epoch)
                writer.add_scalar("BoundaryF1/test", metrics["boundary_f1"], epoch)
            msg += f" dice={metrics['dice']:.4f} bf1={metrics['boundary_f1']:.4f}"
        print(msg, flush=True)
    return {"best": best, "final": metrics}
