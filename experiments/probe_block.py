#!/usr/bin/env python3
"""Diagnostics on what each part of a trained PDC block is actually doing.

    micromamba run -n ml_env python3 experiments/probe_block.py

Goes past "does z move" (that is `analyze_z.py`) and asks which degrees of
freedom earn their parameters, so that improvements can be aimed rather than
guessed. For every checkpoint it reports:

  attn entropy   normalised entropy of the softmax over levels (0 = one level
                 chosen, 1 = uniform). A diffuse distribution means the block is
                 averaging levels rather than selecting one -- which a much
                 cheaper fixed per-level weighting could also do.
  const frac     share of the variance in the attention weights explained by a
                 single global per-level constant. High = the learned "content
                 adaptive" weighting is mostly a fixed reweighting in disguise.
  kp spread      sd of the depth choice across the K^2 kernel points at the same
                 spatial location. ~0 means per-kernel-point z is redundant
                 (consistent with --z-shared being free on dense prediction).
  |dxy|          mean spatial offset magnitude in pixels of the reference level,
                 and the share of sample points displaced by more than one pixel:
                 is the spatial half of the block doing anything?
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_z import CKPTS, parse_tag
from pyrdc.data import build_segmentation_datasets, build_loaders
from pyrdc.models import SmallUNet


def probe(tag, max_batches=2):
    parsed = parse_tag(tag)
    if parsed is None:
        return None
    dataset, head, ch, kw, skips = parsed
    size = (96, 96) if dataset == "kvasir" else (64, 64)

    model = SmallUNet(head, channels=ch, skip_connections=skips, **kw)
    model.load_state_dict(torch.load(os.path.join(CKPTS, tag + ".pth"), map_location="cpu"))
    model.eval()
    # the single-level (DCN-equivalent) head wraps the block one level down
    block = getattr(model.head, "block", model.head)
    L, K2, n_z = block.num_levels or 1, block.K2, block.num_z_channels

    raws = []
    h = block.offset_conv.register_forward_hook(lambda m, i, o: raws.append(o.detach()))
    _, test_ds = build_segmentation_datasets(dataset, size=size)
    _, loader = build_loaders(test_ds, test_ds, batch_size=32, num_workers=4)

    ent, const, spread, dxy, far = [], [], [], [], []
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i >= max_batches:
                break
            raws.clear()
            model(imgs)
            raw = raws[0]
            B, _, H, W = raw.shape

            # --- spatial half -------------------------------------------------
            off = raw[:, : 2 * K2].view(B, 2, K2, H, W)
            mag = off.pow(2).sum(1).sqrt()            # pixels on the reference grid
            dxy.append(mag.mean().item())
            far.append((mag > 1.0).float().mean().item())

            # --- depth half ---------------------------------------------------
            if n_z == 0:                      # DCN-equivalent control: no z branch
                zc = torch.zeros(B, 1, H, W)
                spread.append(0.0)
                continue
            z = raw[:, 2 * K2: 2 * K2 + n_z]
            if block.z_mode == "attention":
                w = torch.softmax(z.view(B, -1, L, H, W), dim=2)   # (B, kp, L, H, W)
                p = w.clamp_min(1e-9)
                ent.append((-(p * p.log()).sum(2) / np.log(L)).mean().item())
                # how much of the weight field is just a global per-level constant?
                mean_w = w.mean(dim=(0, 1, 3, 4)).view(1, 1, L, 1, 1)
                ss_res = (w - mean_w).pow(2).sum().item()
                ss_tot = (w - w.mean()).pow(2).sum().item()
                const.append(1.0 - ss_res / max(ss_tot, 1e-12))
                zc = (w * torch.arange(L, dtype=torch.float32).view(1, 1, L, 1, 1)).sum(2)
                zc = zc / (L - 1)
            else:
                zc = torch.sigmoid(z)                              # (B, kp, H, W)
            if zc.shape[1] > 1:
                spread.append(zc.std(dim=1).mean().item())
            else:
                spread.append(0.0)
    h.remove()

    return {
        "tag": tag, "mode": block.z_mode, "per_kp": zc.shape[1] > 1,
        "entropy": float(np.mean(ent)) if ent else float("nan"),
        "const_frac": float(np.mean(const)) if const else float("nan"),
        "kp_spread": float(np.mean(spread)),
        "dxy": float(np.mean(dxy)), "far": float(np.mean(far)),
    }


ROWS = [
    ("Oxford trilinear+LN", "seg_oxford_pet_pyramidal_zinit=middle_lvlnorm_seed=0"),
    ("Oxford attention", "seg_oxford_pet_pyramidal_attention_zinit=middle_seed=0"),
    ("Oxford attention+LN", "seg_oxford_pet_pyramidal_attention_zinit=middle_lvlnorm_seed=0"),
    ("Oxford attn zshared", "seg_oxford_pet_pyramidal_attention_zinit=middle_zshared_seed=0"),
    ("Oxford single-level", "seg_oxford_pet_pyramidal_single_level_zinit=middle_seed=0"),
    ("Kvasir trilinear+LN", "seg_kvasir_pyramidal_zinit=middle_lvlnorm_seed=0"),
    ("Kvasir attention+LN", "seg_kvasir_pyramidal_attention_zinit=middle_lvlnorm_seed=0"),
    ("Kvasir single-level", "seg_kvasir_pyramidal_single_level_zinit=middle_seed=0"),
]

if __name__ == "__main__":
    print(f"{'config':22s} {'mode':>10s} {'attn ent':>9s} {'const frac':>11s} "
          f"{'kp spread':>10s} {'|dxy| px':>9s} {'>1px':>6s}")
    for label, tag in ROWS:
        if not os.path.exists(os.path.join(CKPTS, tag + ".pth")):
            print(f"{label:22s} (no checkpoint)")
            continue
        r = probe(tag)
        print(f"{label:22s} {r['mode']:>10s} {r['entropy']:9.3f} {r['const_frac']:11.3f} "
              f"{r['kp_spread']:10.4f} {r['dxy']:9.3f} {r['far']:6.3f}")
