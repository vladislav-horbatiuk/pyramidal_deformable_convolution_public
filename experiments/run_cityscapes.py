#!/usr/bin/env python3
"""Cityscapes semantic segmentation runner (19 classes), GPU-oriented.

Designed for a Kaggle P100 / local RTX session; also runs on CPU for smoke
tests. Expects the official extracted layout under --root:
    <root>/leftImg8bit/{train,val}/<city>/*_leftImg8bit.png
    <root>/gtFine/{train,val}/<city>/*_gtFine_labelIds.png

Example (Kaggle):
    python experiments/run_cityscapes.py \
        --root /kaggle/input/cityscapes --head pyramidal_attention \
        --channels 64 --batch-size 16 --epochs 60 --seeds 0
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyrdc.blocks import HEAD_TYPES
from pyrdc.data import (
    ADE20K_CLASS_NAMES, ADE20K_LABEL_MAP, ADE20K_NUM_CLASSES,
    CITYSCAPES_CLASS_NAMES, CITYSCAPES_NUM_CLASSES, Ade20kSeg, CityscapesSeg,
    CityscapesTrainTransform, CityscapesValTransform,
)
from pyrdc.engine import (
    count_parameters, seed_everything, train_segmentation_multiclass,
)
from pyrdc.models import SmallUNet


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="dataset root directory")
    p.add_argument("--dataset", default="cityscapes",
                   choices=["cityscapes", "ade20k"],
                   help="cityscapes (19 classes) or ade20k "
                        "(ADEChallengeData2016 layout, 150 classes)")
    p.add_argument("--head", default="pyramidal_attention", choices=HEAD_TYPES)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--base-size", type=int, nargs=2, default=[1024, 512],
                   help="resize W H applied to all images before crop/eval")
    p.add_argument("--crop", type=int, nargs=2, default=[256, 512],
                   help="train crop H W")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-batch-size", type=int, default=1,
                   help="validation runs on full base-size images, where the "
                        "pyramid heads' sampling tensor is ~4x the train-crop "
                        "size; keep this small on 16GB GPUs")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--z-init", default="middle",
                   choices=["first", "middle", "last", "uniform"])
    p.add_argument("--z-shared", action="store_true")
    p.add_argument("--level-norm", action="store_true")
    p.add_argument("--use-mask", action="store_true")
    p.add_argument("--results-csv", default=None,
                   help="defaults to results/<dataset>.csv")
    p.add_argument("--out-dir", default="results",
                   help="checkpoints/logs directory (use /kaggle/working on Kaggle)")
    p.add_argument("--eval-only", metavar="CKPT",
                   help="skip training: load this state_dict and only write the "
                        "per-class IoU / per-image mIoU dumps for it")
    return p.parse_args()


def head_kwargs(args):
    if not args.head.startswith("pyramidal"):
        return {}
    kw = {"z_init": args.z_init, "use_mask": args.use_mask}
    if args.z_shared:
        kw["z_per_kernel_point"] = False
    if args.level_norm:
        kw["use_level_norm"] = True
    return kw


@torch.no_grad()
def warmup_cuda_handles(device):
    """Force cuBLAS/cuDNN to create their handles while the card is still empty.

    Both libraries allocate their workspaces straight from the driver, *not*
    from PyTorch's caching allocator — and the caching allocator never returns
    memory to the driver. At val resolution (512x1024, ch 64) the pyramid
    sampler reserves several GB before the head's first matmul, so in a fresh
    eval-only process `cublasCreate` finds nothing left and dies with
    CUBLAS_STATUS_ALLOC_FAILED. Training never hit this because its first
    matmul runs in epoch 1, on an empty card."""
    if not str(device).startswith("cuda"):
        return
    torch.mm(torch.zeros(8, 8, device=device), torch.zeros(8, 8, device=device))
    torch.nn.functional.conv2d(
        torch.zeros(1, 1, 8, 8, device=device), torch.zeros(1, 1, 3, 3, device=device)
    )
    torch.cuda.synchronize()


@torch.no_grad()
def dump_val_breakdown(model, val_loader, device, tag, out_dir,
                       num_classes=CITYSCAPES_NUM_CLASSES,
                       class_names=CITYSCAPES_CLASS_NAMES):
    """Write per-class IoU (dataset-level) and per-image mIoU/pixel-acc for the
    trained model. The per-image file is the paired sample for the Wilcoxon
    tests; the per-class file separates frequent from rare classes, which the
    single mIoU number and the pixel accuracy both obscure."""
    from pyrdc.metrics import ConfusionMeter, per_image_miou

    warmup_cuda_handles(device)
    model.eval()
    meter = ConfusionMeter(num_classes)
    rows = []
    for imgs, targets in val_loader:
        logits = model(imgs.to(device))
        targets = targets.to(device)
        meter.update(logits, targets)
        mious = per_image_miou(logits, targets, num_classes).tolist()
        preds = logits.argmax(dim=1)
        valid = targets != 255
        for i, miou in enumerate(mious):
            v = valid[i]
            acc = (preds[i][v] == targets[i][v]).float().mean().item() if v.any() else float("nan")
            rows.append((len(rows), round(miou, 5), round(acc, 5)))

    per_dir = os.path.join(out_dir, "per_image")
    os.makedirs(per_dir, exist_ok=True)
    with open(os.path.join(per_dir, f"{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_idx", "miou", "pixel_acc"])
        w.writerows(rows)

    with open(os.path.join(per_dir, f"{tag}_per_class.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "name", "iou", "gt_pixel_frac"])
        ious = meter.per_class_iou.tolist()
        freqs = meter.class_pixel_freq.tolist()
        for c, (iou, freq) in enumerate(zip(ious, freqs)):
            name = class_names[c] if c < len(class_names) else str(c)
            w.writerow([c, name, round(iou, 5), round(freq, 6)])
    print(f"[{tag}] eval dump: miou={meter.miou:.4f} pixel_acc={meter.pixel_acc:.4f}",
          flush=True)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dataset == "cityscapes":
        ds_cls, label_map = CityscapesSeg, None
        num_classes, class_names = CITYSCAPES_NUM_CLASSES, CITYSCAPES_CLASS_NAMES
    else:
        ds_cls, label_map = Ade20kSeg, ADE20K_LABEL_MAP
        num_classes, class_names = ADE20K_NUM_CLASSES, ADE20K_CLASS_NAMES
    args.results_csv = args.results_csv or f"results/{args.dataset}.csv"
    # eval-only never touches the train split, so don't require it to exist
    train_loader = None
    if not args.eval_only:
        train_ds = ds_cls(
            args.root, "train",
            CityscapesTrainTransform(tuple(args.base_size), tuple(args.crop),
                                     label_map=label_map),
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=True,
            pin_memory=(device == "cuda"),
        )
    val_ds = ds_cls(
        args.root, "val",
        CityscapesValTransform(tuple(args.base_size), label_map=label_map),
    )
    print(f"train={len(train_loader.dataset) if train_loader else 0} "
          f"val={len(val_ds)} device={device}")
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    for seed in args.seeds:
        seed_everything(seed)
        model = SmallUNet(
            args.head, out_channels=num_classes,
            channels=args.channels, **head_kwargs(args),
        )
        n_params = count_parameters(model)
        tag = f"{args.dataset}_{args.head}_ch={args.channels}"
        if args.z_shared:
            tag += "_zshared"
        if args.level_norm:
            tag += "_lvlnorm"
        tag += f"_seed={seed}"

        if args.eval_only:
            model.load_state_dict(torch.load(args.eval_only, map_location="cpu"))
            model.to(device)
            print(f"[{tag}] eval-only from {args.eval_only} params={n_params}", flush=True)
            dump_val_breakdown(model, val_loader, device, tag, args.out_dir,
                               num_classes, class_names)
            continue

        if args.compile:
            model = torch.compile(model)
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=f"{args.out_dir}/tb/{tag}_{int(time.time())}")
        print(f"[{tag}] params={n_params}")
        t0 = time.perf_counter()
        res = train_segmentation_multiclass(
            model, train_loader, val_loader, device, num_classes,
            writer, epochs=args.epochs, lr=args.lr, eval_every=args.eval_every,
        )
        res.update(seed=seed, params=n_params,
                   train_time_s=round(time.perf_counter() - t0, 1))
        print(f"[{tag}] {json.dumps(res)}", flush=True)
        state = getattr(model, "_orig_mod", model).state_dict()
        torch.save(state, f"{args.out_dir}/{tag}.pth")

        os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)
        write_header = not os.path.exists(args.results_csv)
        with open(args.results_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["head", "channels", "options", "seed", "params",
                            "final_miou", "final_pixel_acc",
                            "best_miou", "train_time_s"])
            w.writerow([args.head, args.channels,
                        json.dumps(head_kwargs(args), sort_keys=True), seed,
                        n_params, round(res["final"]["miou"], 4),
                        round(res["final"]["pixel_acc"], 4),
                        round(res["best"]["miou"], 4), res["train_time_s"]])

        # Strictly after the checkpoint and the CSV row: this is post-hoc
        # analysis, and a multi-hour run must not lose its results because the
        # breakdown pass failed. The dump is reproducible later anyway, from
        # the checkpoint via --eval-only.
        try:
            dump_val_breakdown(getattr(model, "_orig_mod", model), val_loader,
                               device, tag, args.out_dir, num_classes, class_names)
        except Exception:
            traceback.print_exc()
            print(f"[{tag}] breakdown dump failed; rerun it later with "
                  f"--eval-only {args.out_dir}/{tag}.pth", flush=True)


if __name__ == "__main__":
    main()
