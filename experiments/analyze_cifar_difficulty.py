#!/usr/bin/env python3
"""CIFAR-100 per-class gains vs class difficulty (paper §where), seed-0 checkpoints.

    micromamba run -n ml_env python3 experiments/analyze_cifar_difficulty.py

Correlating the per-class change (pyramid - control) with the control's own
per-class accuracy is biased negative by regression to the mean: the control's
sampling noise enters both sides with opposite signs. The de-biased estimate
measures difficulty on one random half of each class's test images and the
change on the other half. A single split moves the statistic by ~0.07, so the
split-half numbers are averaged over N_SPLITS seeded per-class splits.
"""
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyrdc.data import build_cifar_datasets, build_loaders
from pyrdc.models import SmallClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPTS = os.path.join(ROOT, "results", "ckpts")
HEAD_KW = dict(z_init="middle", use_mask=False, tanh_spatial_offsets=False,
               coords_as_extra_input=False)
PYRAMID = ("pyramidal 95991", "pyramidal", 16, "cls_cifar100_pyramidal_seed=0.pth")
CONTROLS = [
    ("single ch16 93870", "pyramidal_single_level", 16,
     "cls_cifar100_pyramidal_single_level_seed=0.pth"),
    ("single ch17 97880", "pyramidal_single_level", 17,
     "cls_cifar100_pyramidal_single_level_ch=17_seed=0.pth"),
]
N_SPLITS = 500
N_BINS = 5
HARD, EASY = 0.3, 0.7


def correct(head: str, channels: int, ckpt: str, loader) -> np.ndarray:
    """Per-test-image correctness (bool) of a saved classifier."""
    model = SmallClassifier(head, 100, channels=channels, **HEAD_KW)
    model.load_state_dict(torch.load(os.path.join(CKPTS, ckpt), map_location="cpu"))
    model.eval()
    with torch.no_grad():
        preds = torch.cat([model(x).argmax(1) for x, _ in loader])
    labels = torch.cat([y for _, y in loader])
    return (preds == labels).numpy()


def class_acc(hits: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.array([hits[(y == c) & mask].mean() for c in range(100)])


def half_masks(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random half of every class's test images."""
    a = np.zeros(len(y), bool)
    for c in range(100):
        idx = np.flatnonzero(y == c)
        a[rng.permutation(idx)[: len(idx) // 2]] = True
    return a


def split_half_stats(pyr: np.ndarray, ctl: np.ndarray, y: np.ndarray) -> dict:
    rng = np.random.default_rng(0)
    null, debiased, bins = [], [], []
    for _ in range(N_SPLITS):
        a = half_masks(y, rng)
        b = ~a
        ctl_a, ctl_b = class_acc(ctl, y, a), class_acc(ctl, y, b)
        delta_b = class_acc(pyr, y, b) - ctl_b
        null.append(spearmanr(ctl_a, ctl_b - ctl_a)[0])
        debiased.append(spearmanr(ctl_a, delta_b)[0])
        edges = np.quantile(ctl_a, np.linspace(0, 1, N_BINS + 1))[1:-1]
        which = np.digitize(ctl_a, edges)
        bins.append([delta_b[which == i].mean() for i in range(N_BINS)])
    bins = np.array(bins)
    return dict(null=np.array(null), debiased=np.array(debiased),
                bins=bins.mean(0), all_bins_positive=(bins > 0).all(1).mean())


def report(name: str, pyr: np.ndarray, ctl: np.ndarray, y: np.ndarray) -> None:
    every = np.ones(len(y), bool)
    acc_c = class_acc(ctl, y, every)
    delta = class_acc(pyr, y, every) - acc_c
    rho, p = spearmanr(acc_c, delta)
    s = split_half_stats(pyr, ctl, y)
    print(f"\n=== pyramid vs {name} (seed 0) ===")
    print(f"  accuracy: pyramid {pyr.mean():.4f}  control {ctl.mean():.4f}  "
          f"Δ {pyr.mean() - ctl.mean():+.4f}")
    print(f"  naive ρ(control acc, Δ) = {rho:+.3f} (p={p:.3g}); "
          f"hard (<{HARD}) Δ {delta[acc_c < HARD].mean():+.3f} "
          f"[{(acc_c < HARD).sum()} classes], easy (≥{EASY}) "
          f"Δ {delta[acc_c >= EASY].mean():+.3f} [{(acc_c >= EASY).sum()} classes]")
    print(f"  null ρ (control vs own split-half change): "
          f"{s['null'].mean():+.3f} ± {s['null'].std():.3f}")
    print(f"  de-biased ρ: {s['debiased'].mean():+.3f} ± {s['debiased'].std():.3f}; "
          f"significant (|ρ|>0.197, p<.05) in "
          f"{(np.abs(s['debiased']) > 0.197).mean():.0%} of splits")
    print(f"  de-biased Δ by difficulty quintile, hardest→easiest: "
          + " ".join(f"{v:+.3f}" for v in s["bins"])
          + f"; every bin > 0 in {s['all_bins_positive']:.0%} of splits")


def main() -> None:
    torch.set_num_threads(8)
    _, test_ds, _ = build_cifar_datasets("cifar100")
    _, loader = build_loaders(test_ds, test_ds, batch_size=500, num_workers=2)
    y = np.array(test_ds.targets)
    pyr = correct(*PYRAMID[1:], loader)
    for name, head, ch, ckpt in CONTROLS:
        if not os.path.exists(os.path.join(CKPTS, ckpt)):
            print(f"\n(skipping {name}: no checkpoint {ckpt})")
            continue
        report(name, pyr, correct(head, ch, ckpt, loader), y)


if __name__ == "__main__":
    main()
