#!/usr/bin/env python3
"""Classification experiment runner (CIFAR-10/100).

Example:
    micromamba run -n ml_env python3 experiments/run_cls.py \
        --dataset cifar10 --head pyramidal_attention --seeds 0 1 2 --epochs 100
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
from pyrdc.data import build_cifar_datasets, build_loaders
from pyrdc.engine import (
    seed_everything, count_parameters, train_classification,
)
from pyrdc.models import SmallClassifier


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--head", default="pyramidal", choices=HEAD_TYPES)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--z-init", default="middle",
                   choices=["first", "middle", "last", "uniform"])
    p.add_argument("--use-mask", action="store_true")
    p.add_argument("--tanh-offsets", action="store_true")
    p.add_argument("--coords-input", action="store_true")
    p.add_argument("--no-adapters", action="store_true")
    p.add_argument("--z-shared", action="store_true")
    p.add_argument("--level-norm", action="store_true")
    p.add_argument("--residual-cross-scale", action="store_true",
                   help="out = DCN(reference) + g*CrossScale(pyramid), g init 0")
    p.add_argument("--z-weight-init-std", type=float, default=0.0,
                   help="random init std for the depth rows of offset_conv")
    p.add_argument("--save-models", action="store_true",
                   help="save final state_dict to results/ckpts/<tag>.pth")
    p.add_argument("--results-csv", default="results/classification.csv")
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
    if args.z_shared:
        kw["z_per_kernel_point"] = False
    if args.level_norm:
        kw["use_level_norm"] = True
    if args.residual_cross_scale:
        kw["residual_cross_scale"] = True
    if args.z_weight_init_std:
        kw["z_weight_init_std"] = args.z_weight_init_std
    return kw


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, test_ds, num_classes = build_cifar_datasets(args.dataset)
    train_loader, test_loader = build_loaders(
        train_ds, test_ds, batch_size=args.batch_size, num_workers=args.num_workers
    )

    all_results = []
    for seed in args.seeds:
        seed_everything(seed)
        model = SmallClassifier(
            args.head, num_classes, channels=args.channels,
            use_adapters=not args.no_adapters, **head_kwargs(args),
        )
        n_params = count_parameters(model)
        if args.compile:
            model = torch.compile(model)
        tag = f"cls_{args.dataset}_{args.head}"
        if args.z_shared:
            tag += "_zshared"
        if args.level_norm:
            tag += "_lvlnorm"
        if args.residual_cross_scale:
            tag += "_rescross"
        if args.z_weight_init_std:
            tag += f"_zwstd={args.z_weight_init_std:g}"
        if args.channels != 16:
            tag += f"_ch={args.channels}"
        if args.weight_decay != 1e-5:
            tag += f"_wd={args.weight_decay:g}"
        if args.epochs != 150:  # 150 = the study's locked schedule (RESULTS §2)
            tag += f"_ep={args.epochs}"
        tag += f"_seed={seed}"
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=f"{args.logdir}/{tag}_{int(time.time())}")
        print(f"[{tag}] params={n_params} device={device}")
        t0 = time.perf_counter()
        res = train_classification(
            model, train_loader, test_loader, device, writer,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        )
        elapsed = time.perf_counter() - t0
        res.update(seed=seed, params=n_params, train_time_s=round(elapsed, 1))
        print(f"[{tag}] {json.dumps(res)}")
        if args.save_models:
            os.makedirs("results/ckpts", exist_ok=True)
            state = getattr(model, "_orig_mod", model).state_dict()
            torch.save(state, f"results/ckpts/{tag}.pth")
        all_results.append(res)

    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
    write_header = not os.path.exists(args.results_csv)
    with open(args.results_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["dataset", "head", "options", "seed", "params",
                        "final_acc", "best_acc", "train_time_s"])
        csv_opts = dict(head_kwargs(args))
        if args.weight_decay != 1e-5:
            csv_opts["weight_decay"] = args.weight_decay
        if args.epochs != 150:
            csv_opts["epochs"] = args.epochs
        for res in all_results:
            w.writerow([
                args.dataset, args.head,
                json.dumps(csv_opts, sort_keys=True), res["seed"],
                res["params"], round(res["final_acc"], 4),
                round(res["best_acc"], 4), res["train_time_s"],
            ])
    accs = [r["final_acc"] for r in all_results]
    mean = sum(accs) / len(accs)
    std = (sum((a - mean) ** 2 for a in accs) / max(len(accs) - 1, 1)) ** 0.5
    print(f"final acc over {len(accs)} seed(s): {mean:.4f} +- {std:.4f}")


if __name__ == "__main__":
    main()
