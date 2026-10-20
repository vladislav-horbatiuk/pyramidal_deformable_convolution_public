#!/usr/bin/env python3
"""Is the Cityscapes redistribution a decision-threshold (class-prior) effect?

    micromamba run -n ml_env python3 experiments/analyze_prior_shift.py [--rerun]

Stage 1 (cached in results/prior_shift_sweep.json; --rerun to redo, ~20 min
CPU): evaluate the FROZEN un-normalised control (single ch66, seed 0) on the
Cityscapes val set under an inference-time logit adjustment
    logits_c -> logits_c - lam * log(prior_c)
for lam in {0, .05, .1, .2, .4, .8} — the long-tail logit-adjustment knob
(Menon et al., 2021). No training, no architecture change.

Stage 2: compare each lam's per-class IoU delta (vs lam=0) with the OBSERVED
delta of the normalised control (3 seeds) vs the un-normalised control
(4 seeds). If a small lam reproduces the observed pattern, the redistribution
is mechanically a class-prior/threshold shift, not representation learning.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CACHE = os.path.join(ROOT, "results", "prior_shift_sweep.json")
LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]


def run_sweep():
    import torch
    from pyrdc.data import CityscapesSeg, CityscapesValTransform
    from pyrdc.models import SmallUNet

    pc = pd.read_csv(os.path.join(
        ROOT, "kaggle_artifacts", "per_image",
        "cityscapes_pyramidal_single_level_ch=66_seed=0_per_class.csv"))
    prior = pc["gt_pixel_frac"].values.astype(np.float64)
    logprior = torch.tensor(np.log(np.maximum(prior, 1e-8)), dtype=torch.float32)

    model = SmallUNet("pyramidal_single_level", out_channels=19, channels=66,
                      z_init="middle")
    model.load_state_dict(torch.load(os.path.join(
        ROOT, "kaggle_artifacts",
        "cityscapes_pyramidal_single_level_ch=66_seed=0.pth"), map_location="cpu"))
    model.eval()
    ds = CityscapesSeg(os.path.join(ROOT, "cityscapes"), "val",
                       CityscapesValTransform((1024, 512)))
    conf = {l: np.zeros((19, 19), dtype=np.int64) for l in LAMBDAS}
    with torch.no_grad():
        for i in range(len(ds)):
            img, tgt = ds[i]
            logits = model(img.unsqueeze(0))[0]
            valid = tgt != 255
            t = tgt[valid].numpy()
            for lam in LAMBDAS:
                pred = (logits - lam * logprior.view(-1, 1, 1)).argmax(0)[valid].numpy()
                np.add.at(conf[lam], (t, pred), 1)
            if i % 100 == 0:
                print(f"{i}/{len(ds)}", flush=True)
    out = {}
    for lam, c in conf.items():
        inter = np.diag(c).astype(float)
        union = c.sum(0) + c.sum(1) - np.diag(c)
        iou = inter / np.maximum(union, 1)
        out[str(lam)] = {"miou": float(iou.mean()),
                         "pixacc": float(np.diag(c).sum() / c.sum()),
                         "iou": iou.tolist()}
    json.dump({"prior": prior.tolist(), "names": pc["name"].tolist(),
               "sweep": out}, open(CACHE, "w"))


def _seedavg(pattern):
    files = sorted(
        glob.glob(os.path.join(ROOT, "kaggle_artifacts", "per_image", pattern))
        + glob.glob(os.path.join(ROOT, "results", "cityscapes_local",
                                 "per_image", pattern)))
    return (np.mean([pd.read_csv(f).sort_values("class_id")["iou"].values
                     for f in files], axis=0), len(files))


def compare():
    d = json.load(open(CACHE))
    prior = np.array(d["prior"])
    freq = prior >= np.median(prior)
    base = np.array(d["sweep"]["0.0"]["iou"])
    ln, n1 = _seedavg("cityscapes_pyramidal_single_level_ch=66_lvlnorm_seed=*_per_class.csv")
    un, n2 = _seedavg("cityscapes_pyramidal_single_level_ch=66_seed=?_per_class.csv")
    obs = ln - un
    print(f"observed LN-control delta (n={n1} vs n={n2}): "
          f"rho(freq)={spearmanr(prior, obs).statistic:+.3f}  "
          f"rare={obs[~freq].mean():+.4f}  freq={obs[freq].mean():+.4f}")
    print(f"{'lam':>5} {'mIoU':>7} {'pixacc':>7} {'rho(freq,D)':>11} "
          f"{'rare D':>8} {'freq D':>8} {'corr w/ obs':>11}")
    for lam in map(str, LAMBDAS):
        s = d["sweep"][lam]
        delta = np.array(s["iou"]) - base
        rho = spearmanr(prior, delta).statistic if lam != "0.0" else 0.0
        corr = spearmanr(delta, obs).statistic if lam != "0.0" else float("nan")
        print(f"{lam:>5} {s['miou']:7.4f} {s['pixacc']:7.4f} {rho:11.3f} "
              f"{delta[~freq].mean():+8.4f} {delta[freq].mean():+8.4f} {corr:11.3f}")


if __name__ == "__main__":
    if "--rerun" in sys.argv or not os.path.exists(CACHE):
        run_sweep()
    compare()
