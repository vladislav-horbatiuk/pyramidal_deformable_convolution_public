#!/usr/bin/env python3
"""Where does the Oxford parity win live? Boundary-F1 + mask-size strata.

    micromamba run -n ml_env python3 experiments/analyze_where.py

Two views on full-z+LN (19 420) vs spatial-only+LN ch17 (19 195), n=8 each:

1. Boundary-F1 — already computed by every training run and logged to
   TensorBoard (`BoundaryF1/test`); read the final value per seed and test
   across seeds. Cross-scale sampling was motivated by boundaries and thin
   structures, so this is the metric it should move.
2. Mask-size strata — join the existing per-image Dice dumps with per-image
   mask areas (dataset pass, no model) and compare the seed-averaged Δ per
   area quintile. No new forward passes anywhere.
"""
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel, wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
TB = os.path.join(ROOT, "tensorboard_logs")
PER_IMAGE = os.path.join(ROOT, "results", "per_image")

PAIRS = [
    ("OXFORD full-z+LN vs single+LN ch17 (the parity win)",
     "oxford_pet",
     "seg_oxford_pet_pyramidal_zinit=middle_lvlnorm",
     "seg_oxford_pet_pyramidal_single_level_zinit=middle_lvlnorm_ch=17"),
    ("OXFORD cheap-z vs single+LN ch17",
     "oxford_pet",
     "seg_oxford_pet_pyramidal_zinit=middle_zshared_lvlnorm",
     "seg_oxford_pet_pyramidal_single_level_zinit=middle_lvlnorm_ch=17"),
    ("KVASIR cheap-z vs single+LN ch17",
     "kvasir",
     "seg_kvasir_pyramidal_zinit=middle_zshared_lvlnorm",
     "seg_kvasir_pyramidal_single_level_zinit=middle_lvlnorm_ch=17"),
]


def tb_runs():
    out = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(TB, "*"))):
        out[re.sub(r"_\d{10}$", "", os.path.basename(path))].append(path)
    return out


def final_scalar(run_dir, tag):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 200000})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None
    return ea.Scalars(tag)[-1].value


def per_seed_bf1(runs, tag, seeds=range(10)):
    out = {}
    for s in seeds:
        key = f"{tag}_seed={s}"
        if key in runs:
            v = final_scalar(sorted(runs[key])[-1], "BoundaryF1/test")
            if v is not None:
                out[s] = v
    return out


def mask_areas(dataset_name):
    """Foreground fraction per test image, in loader order (shuffle=False)."""
    from pyrdc.data import build_segmentation_datasets
    size = (96, 96) if dataset_name == "kvasir" else (64, 64)
    _, test_ds = build_segmentation_datasets(dataset_name, size=size)
    return np.array([float(test_ds[i][1].float().mean()) for i in range(len(test_ds))])


def seed_mean_dice(tag, seeds=range(10)):
    per_seed = []
    for s in seeds:
        p = os.path.join(PER_IMAGE, f"{tag}_seed={s}.csv")
        if os.path.exists(p):
            per_seed.append(pd.read_csv(p).sort_values("image_idx")["dice"].values)
    return np.mean(per_seed, axis=0), len(per_seed)


def main():
    runs = tb_runs()
    for title, dataset, tag_a, tag_b in PAIRS:
        print(f"\n===== {title} =====")
        a, b = per_seed_bf1(runs, tag_a), per_seed_bf1(runs, tag_b)
        common = sorted(set(a) & set(b))
        av, bv = np.array(list(a.values())), np.array(list(b.values()))
        _, pw = ttest_ind(av, bv, equal_var=False)
        pp = (ttest_rel([a[s] for s in common], [b[s] for s in common])[1]
              if len(common) > 1 else float("nan"))
        print(f"  boundary-F1: arm {av.mean():.4f}±{av.std():.4f} (n={len(av)})  "
              f"ctrl {bv.mean():.4f}±{bv.std():.4f} (n={len(bv)})  "
              f"D={av.mean() - bv.mean():+.4f}  welch_p={pw:.3f}  "
              f"paired_p={pp:.3f} (n={len(common)})")

        da, na = seed_mean_dice(tag_a)
        db, nb = seed_mean_dice(tag_b)
        areas = mask_areas(dataset)
        assert len(areas) == len(da), (len(areas), len(da))
        q = pd.qcut(areas, 5, labels=False)
        print(f"  per-image Dice delta by mask-area quintile "
              f"(seed-avg over {na}/{nb} seeds):")
        for k in range(5):
            m = q == k
            d = da[m] - db[m]
            _, p = wilcoxon(da[m], db[m])
            print(f"    Q{k + 1} (area {areas[m].min():.3f}-{areas[m].max():.3f}, "
                  f"n={m.sum():4d}): D={d.mean():+.4f}  p={p:.2e}")


if __name__ == "__main__":
    main()
