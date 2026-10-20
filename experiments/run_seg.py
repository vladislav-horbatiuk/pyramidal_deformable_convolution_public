#!/usr/bin/env python3
"""Segmentation experiment runner.

Replaces the copy-pasted notebook training cells with one reproducible CLI.
Every run is tagged with head type, options and seed; results (mean over seeds)
are appended to a CSV so tables for the paper can be generated directly.

Example:
    micromamba run -n ml_env python3 experiments/run_seg.py \
        --dataset oxford_pet --head pyramidal --seeds 0 1 2 --epochs 60
"""
import argparse
import csv
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyrdc.blocks import HEAD_TYPES
from pyrdc.data import build_segmentation_datasets, build_loaders
from pyrdc.engine import (
    seed_everything, count_parameters, train_segmentation,
)
from pyrdc.models import SmallUNet


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="oxford_pet",
                   choices=["oxford_pet", "sbd", "urbansys", "kvasir"])
    p.add_argument("--head", default="pyramidal", choices=HEAD_TYPES)
    p.add_argument("--size", type=int, nargs=2, default=[64, 64],
                   help="input H W")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5,
                   help="Adam weight decay; raise it to test whether the "
                        "pyramidal heads' larger train/val gap is fixable by "
                        "regularisation rather than by capacity")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--compile", action="store_true")
    # head options (pyramidal variants)
    p.add_argument("--z-init", default="middle",
                   choices=["first", "middle", "last", "uniform"])
    p.add_argument("--use-mask", action="store_true")
    p.add_argument("--tanh-offsets", action="store_true")
    p.add_argument("--coords-input", action="store_true")
    p.add_argument("--no-adapters", action="store_true")
    p.add_argument("--freeze-z", action="store_true",
                   help="freeze the z/attention branch at its init value")
    p.add_argument("--z-shared", action="store_true",
                   help="share one z/attention prediction across all K*K kernel "
                        "points (cheap-z; pyramidal/attention heads only)")
    p.add_argument("--level-norm", action="store_true",
                   help="per-level GroupNorm on the pyramid before sampling")
    p.add_argument("--residual-cross-scale", action="store_true",
                   help="out = DCN(reference) + g*CrossScale(pyramid) with g "
                        "initialised to 0, so the block starts as an exact DCN "
                        "and adds cross-scale context instead of trading the "
                        "reference level away for it")
    p.add_argument("--offset-scale-by-level", action="store_true",
                   help="scale spatial offsets by the stride of the sampled "
                        "level (trilinear z only)")
    p.add_argument("--z-stretch", type=float, default=0.0,
                   help="stretch-and-clip the depth sigmoid by this eps so the "
                        "first/last levels are reachable exactly (trilinear z only)")
    p.add_argument("--offset-lr-mult", type=float, default=1.0,
                   help="LR multiplier for the offset branch (DCN practice: 0.1)")
    p.add_argument("--offset-no-decay", action="store_true",
                   help="exclude the offset branch from weight decay")
    p.add_argument("--offset-mode", default="free",
                   choices=["free", "affine", "similarity"],
                   help="how the K^2 spatial offsets are generated: 'free' = one "
                        "per kernel point (2*K^2 channels); 'affine' = t + A p_k "
                        "(6); 'similarity' = scale*rotation + t (4)")
    p.add_argument("--z-weight-init-std", type=float, default=0.0,
                   help="random init std for the depth rows of offset_conv "
                        "(0 = the zero-init default)")
    p.add_argument("--no-skips", action="store_true",
                   help="build the U-Net without skip connections")
    p.add_argument("--dump-per-image", action="store_true",
                   help="write per-test-image dice to results/per_image/{tag}.csv")
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--save-models", action="store_true",
                   help="save final state_dict per run under results/ckpts/")
    p.add_argument("--results-csv", default="results/segmentation.csv")
    p.add_argument("--logdir", default="tensorboard_logs")
    return p.parse_args()


def head_kwargs(args):
    if not args.head.startswith("pyramidal"):
        return {}
    kw = {
        "z_init": args.z_init,
        "use_mask": args.use_mask,
        "tanh_spatial_offsets": args.tanh_offsets,
        "coords_as_extra_input": args.coords_input,
    }
    if args.freeze_z:
        kw["freeze_z"] = True  # only PyramidalDeformConv-family heads accept it
    if args.z_shared:
        kw["z_per_kernel_point"] = False  # not supported by reduced heads
    if args.level_norm:
        kw["use_level_norm"] = True  # reduced heads accept this since iter 7
    if args.residual_cross_scale:
        kw["residual_cross_scale"] = True  # full PyramidalDeformConv only
    if args.z_weight_init_std:
        kw["z_weight_init_std"] = args.z_weight_init_std
    if args.offset_mode != "free":
        kw["offset_mode"] = args.offset_mode
    if args.offset_scale_by_level:
        kw["scale_offsets_by_level"] = True  # full PyramidalDeformConv only
    if args.z_stretch:
        kw["z_stretch"] = args.z_stretch  # full PyramidalDeformConv only
    return kw


def run_tag(args, seed):
    opts = []
    if args.head.startswith("pyramidal"):
        opts.append(f"zinit={args.z_init}")
        if args.use_mask:
            opts.append("mask")
        if args.tanh_offsets:
            opts.append("tanh")
        if args.coords_input:
            opts.append("coords")
        if args.freeze_z:
            opts.append("freezez")
        if args.z_shared:
            opts.append("zshared")
        if args.level_norm:
            opts.append("lvlnorm")
        if args.residual_cross_scale:
            opts.append("rescross")
        if args.z_weight_init_std:
            opts.append(f"zwstd={args.z_weight_init_std:g}")
        if args.offset_mode != "free":
            opts.append(args.offset_mode)
        if args.offset_scale_by_level:
            opts.append("offscale")
        if args.z_stretch:
            opts.append(f"zstretch={args.z_stretch:g}")
        if args.offset_lr_mult != 1.0:
            opts.append(f"offlr={args.offset_lr_mult:g}")
        if args.offset_no_decay:
            opts.append("offnowd")
        if args.no_adapters:
            opts.append("noadapt")
    if args.no_skips:
        opts.append("noskip")
    if args.channels != 16:
        opts.append(f"ch={args.channels}")
    if args.weight_decay != 1e-5:
        opts.append(f"wd={args.weight_decay:g}")
    opts.append(f"seed={seed}")
    return f"seg_{args.dataset}_{args.head}_" + "_".join(opts)


def append_csv_row(args, res):
    """Append one finished seed to the results CSV."""
    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
    write_header = not os.path.exists(args.results_csv)
    # training params are not head kwargs, but they must appear in the options
    # column or two runs differing only in them collapse to identical rows
    # (the iter-16 offset-lr rows were written without this and had to be
    # repaired by file position — don't repeat that)
    csv_opts = dict(head_kwargs(args))
    if args.weight_decay != 1e-5:
        csv_opts["weight_decay"] = args.weight_decay
    if args.offset_lr_mult != 1.0:
        csv_opts["offset_lr_mult"] = args.offset_lr_mult
    if args.offset_no_decay:
        csv_opts["offset_no_decay"] = True
    with open(args.results_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["dataset", "head", "options", "seed", "params",
                        "final_dice", "final_iou", "final_bf1",
                        "best_dice", "best_iou", "best_bf1", "train_time_s"])
        w.writerow([
            args.dataset, args.head,
            json.dumps(csv_opts, sort_keys=True), res["seed"], res["params"],
            round(res["final"]["dice"], 4), round(res["final"]["iou"], 4),
            round(res["final"]["boundary_f1"], 4),
            round(res["best"]["dice"], 4), round(res["best"]["iou"], 4),
            round(res["best"]["boundary_f1"], 4), res["train_time_s"],
        ])


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, test_ds = build_segmentation_datasets(args.dataset, size=tuple(args.size))
    train_loader, test_loader = build_loaders(
        train_ds, test_ds, batch_size=args.batch_size, num_workers=args.num_workers
    )

    all_results = []
    for seed in args.seeds:
        seed_everything(seed)
        model = SmallUNet(
            args.head, channels=args.channels, use_adapters=not args.no_adapters,
            skip_connections=not args.no_skips, **head_kwargs(args),
        )
        n_params = count_parameters(model)
        if args.compile:
            model = torch.compile(model)
        tag = run_tag(args, seed)
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=f"{args.logdir}/{tag}_{int(time.time())}")
        print(f"[{tag}] params={n_params} device={device}")
        t0 = time.perf_counter()
        res = train_segmentation(
            model, train_loader, test_loader, device, writer,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            eval_every=args.eval_every, offset_lr_mult=args.offset_lr_mult,
            offset_no_decay=args.offset_no_decay,
        )
        elapsed = time.perf_counter() - t0
        res.update(seed=seed, params=n_params, train_time_s=round(elapsed, 1))
        print(f"[{tag}] {json.dumps(res)}", flush=True)
        if args.save_models:
            os.makedirs("results/ckpts", exist_ok=True)
            state = getattr(model, "_orig_mod", model).state_dict()
            torch.save(state, f"results/ckpts/{tag}.pth")
        if args.dump_per_image:
            from pyrdc.metrics import per_image_dice

            os.makedirs("results/per_image", exist_ok=True)
            model.eval()
            dices = []
            with torch.no_grad():
                for imgs, masks in test_loader:  # shuffle=False -> stable order
                    logits = model(imgs.to(device))
                    dices.extend(per_image_dice(logits, masks.to(device)).tolist())
            with open(f"results/per_image/{tag}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["image_idx", "dice"])
                w.writerows((i, round(d, 5)) for i, d in enumerate(dices))
        all_results.append(res)
        # Flush this seed's row now rather than after the whole seed loop. A
        # power loss during iteration 10 killed a 5-seed run at seed 7 and threw
        # away the two completed seeds, whose per-image dumps were already on
        # disk -- leaving dumps with no matching CSV rows, which silently skews
        # any analysis that averages dumps over seeds.
        append_csv_row(args, res)

    dices = [r["final"]["dice"] for r in all_results]
    mean = sum(dices) / len(dices)
    std = (sum((d - mean) ** 2 for d in dices) / max(len(dices) - 1, 1)) ** 0.5
    print(f"final dice over {len(dices)} seed(s): {mean:.4f} +- {std:.4f}")


if __name__ == "__main__":
    main()
