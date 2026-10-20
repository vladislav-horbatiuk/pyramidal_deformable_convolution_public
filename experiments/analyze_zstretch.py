#!/usr/bin/env python3
"""Depth stretch-and-clip vs plain sigmoid (iterations 20-21), cheap-z + LN.

    micromamba run -n ml_env python3 experiments/analyze_zstretch.py

Pairs every stretch run with the baseline run of the same seed and dataset
(identical flags otherwise) and reports dataset Dice, per-image Dice, the
training-loss tail, and where the trained depth branch puts its sampling weight.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_ind, ttest_rel, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import analyze as A  # noqa: E402
import analyze_z as Z  # noqa: E402
from pyrdc.data import build_loaders, build_segmentation_datasets  # noqa: E402
from pyrdc.models import SmallUNet  # noqa: E402

SEEDS = range(5)
DATASETS = ("kvasir", "oxford_pet")
BASE = "seg_{ds}_pyramidal_zinit=middle_zshared_lvlnorm"
ARM = BASE + "_zstretch=0.1"
PROBE_BATCHES = 6


def dataset_dice(ds: str, stretch: bool) -> np.ndarray:
    """Final dataset-level Dice per seed, from results/segmentation.csv."""
    d = pd.read_csv(os.path.join(ROOT, "results", "segmentation.csv"))
    d = d[(d["dataset"] == ds) & (d["head"] == "pyramidal")]
    want = {"coords_as_extra_input": False, "tanh_spatial_offsets": False,
            "use_level_norm": True, "use_mask": False, "z_init": "middle",
            "z_per_kernel_point": False}
    if stretch:
        want["z_stretch"] = 0.1
    d = d[d["options"].apply(lambda o: json.loads(o) == want)]
    d = d.drop_duplicates("seed", keep="last").set_index("seed")
    return d.loc[list(SEEDS), "final_dice"].to_numpy()


def per_image_dice(tag: str) -> pd.Series:
    return pd.read_csv(os.path.join(ROOT, "results", "per_image", f"{tag}.csv")).set_index(
        "image_idx")["dice"]


def train_tail(runs: dict, tag: str) -> float:
    v = A._scalars(sorted(runs[tag])[-1])["epoch_loss/train"]
    return float(v[-max(1, int(len(v) * 0.1)):].mean())


def level_mass(tag: str) -> np.ndarray:
    """Share of sampling weight per pyramid level on the first test batches."""
    dataset, head, ch, kw, skips = Z.parse_tag(tag)
    size = (96, 96) if dataset == "kvasir" else (64, 64)
    model = SmallUNet(head, channels=ch, skip_connections=skips, **kw)
    model.load_state_dict(torch.load(os.path.join(ROOT, "results", "ckpts", tag + ".pth"),
                                     map_location="cpu"))
    model.eval()
    blk = model.head
    lo_ch = blk.n_xy
    grabbed = []
    hook = blk.offset_conv.register_forward_hook(
        lambda _m, _i, out: grabbed.append(out[:, lo_ch: lo_ch + blk.num_z_channels]))
    _, test_ds = build_segmentation_datasets(dataset, size=size)
    _, loader = build_loaders(test_ds, test_ds, batch_size=128, num_workers=4)
    raws = []
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i >= PROBE_BATCHES:
                break
            grabbed.clear()
            model(imgs)
            raws.append(grabbed[0].flatten())
    hook.remove()
    idx = blk.depth(torch.cat(raws)) * (blk.num_levels - 1)
    lo = idx.floor().long().clamp(0, blk.num_levels - 1)
    hi = (lo + 1).clamp(max=blk.num_levels - 1)
    frac = idx - lo.float()
    mass = torch.zeros(blk.num_levels)
    mass.index_add_(0, lo, 1 - frac)
    mass.index_add_(0, hi, frac)
    return (mass / mass.sum()).numpy()


def report(ds: str, runs: dict) -> None:
    base, arm = BASE.format(ds=ds), ARM.format(ds=ds)
    print(f"\n===== {ds}: stretch (eps=0.1) minus plain sigmoid, seeds 0-4 =====")
    db, da = dataset_dice(ds, False), dataset_dice(ds, True)
    print(f"dataset Dice   base {np.round(db, 4)}  stretch {np.round(da, 4)}")
    print(f"  delta {da.mean() - db.mean():+.4f}  Welch p={ttest_ind(da, db, equal_var=False).pvalue:.3g}"
          f"  paired p={ttest_rel(da, db).pvalue:.3g}  seeds better {np.sum(da > db)}/5")
    pb = pd.concat([per_image_dice(f"{base}_seed={s}") for s in SEEDS], axis=1)
    pa = pd.concat([per_image_dice(f"{arm}_seed={s}") for s in SEEDS], axis=1)
    mb, ma = pb.mean().to_numpy(), pa.mean().to_numpy()
    img = pa.mean(axis=1) - pb.mean(axis=1)
    print(f"per-image Dice delta {ma.mean() - mb.mean():+.4f}  seed paired p={ttest_rel(ma, mb).pvalue:.3g}"
          f"  image-level Wilcoxon p={wilcoxon(img).pvalue:.3g} ({np.mean(img > 0):.0%} of images better)")
    tb = np.array([train_tail(runs, f"{base}_seed={s}") for s in SEEDS])
    ta = np.array([train_tail(runs, f"{arm}_seed={s}") for s in SEEDS])
    print(f"train-loss tail delta {ta.mean() - tb.mean():+.4f}  paired p={ttest_rel(ta, tb).pvalue:.3g}"
          f"  seeds lower {np.sum(ta < tb)}/5")
    lb = np.mean([level_mass(f"{base}_seed={s}") for s in SEEDS], axis=0)
    la = np.mean([level_mass(f"{arm}_seed={s}") for s in SEEDS], axis=0)
    print(f"level mass p1/b/d2/d1  base {np.round(lb, 3)}  stretch {np.round(la, 3)}")


def main() -> None:
    runs = A._tb_runs(A.TB)
    for ds in DATASETS:
        report(ds, runs)


if __name__ == "__main__":
    main()
