#!/usr/bin/env python3
"""What does the depth branch actually predict?

    micromamba run -n ml_env python3 experiments/analyze_z.py [--figure]

Hooks `head.offset_conv` on trained checkpoints, replays the test set, and
recovers the depth decision exactly as the forward pass computes it:
`sigmoid(z_raw)` for z_mode='trilinear', `softmax(logits, dim=levels)` for
z_mode='attention'. Reports, per configuration:

  mean z        where the block samples on average. The SmallUNet pyramid is
                [p1, b, d2, d1] with d1 the full-resolution reference level, so
                z=1 means "behave like DCN on the reference level" and z=0 means
                "reach all the way into the coarsest level".
  at reference  fraction of sample points with z > 0.9 -- the collapse check. A
                block that learned to ignore the pyramid would score ~1.0 here.
  spatial sd    sd of z across positions within an image, averaged over images:
                is the depth choice content-adaptive or a learned constant?
  image sd      sd of the per-image mean z across images.

The comparison that matters is learned vs frozen (does training move z at all?)
and level-norm vs none (does normalising the levels change *what* the block
does, or only how well it works?).
"""
import argparse
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyrdc.data import build_segmentation_datasets, build_loaders
from pyrdc.models import SmallUNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPTS = os.path.join(ROOT, "results", "ckpts")
LEVEL_NAMES = ["p1 (coarsest)", "b", "d2", "d1 (reference)"]


def parse_tag(tag):
    """Rebuild the model config from a run tag."""
    # longest head name first: 'pyramidal' is a prefix of the others
    m = re.match(r"seg_(oxford_pet|kvasir)_"
                 r"(pyramidal_single_level|pyramidal_attention|pyramidal)_", tag)
    if not m:
        return None
    dataset, head = m.group(1), m.group(2)
    ch = int(re.search(r"_ch=(\d+)", tag).group(1)) if "_ch=" in tag else 16
    kw = {
        "z_init": "middle" if "zinit=middle" in tag else "last",
        "use_mask": False,
        "tanh_spatial_offsets": False,
        "coords_as_extra_input": False,
    }
    if "_lvlnorm" in tag:
        kw["use_level_norm"] = True
    if "_zshared" in tag:
        kw["z_per_kernel_point"] = False
    if "_freezez" in tag:
        kw["freeze_z"] = True
    m = re.search(r"_zstretch=([0-9.]+)", tag)
    if m:
        kw["z_stretch"] = float(m.group(1))
    return dataset, head, ch, kw, ("_noskip" not in tag)


def z_stats(tag, max_batches=None):
    parsed = parse_tag(tag)
    if parsed is None:
        return None
    dataset, head, ch, kw, skips = parsed
    size = (96, 96) if dataset == "kvasir" else (64, 64)
    batch = 64 if dataset == "kvasir" else 128

    model = SmallUNet(head, channels=ch, skip_connections=skips, **kw)
    state = torch.load(os.path.join(CKPTS, tag + ".pth"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    block = model.head
    L, K2 = block.num_levels, block.K2
    n_z = block.num_z_channels
    grabbed = []

    def hook(_mod, _inp, out):
        # exactly the slice the forward pass takes
        grabbed.append(out[:, 2 * K2: 2 * K2 + n_z].detach())

    handle = block.offset_conv.register_forward_hook(hook)

    _, test_ds = build_segmentation_datasets(dataset, size=size)
    _, loader = build_loaders(test_ds, test_ds, batch_size=batch, num_workers=4)

    per_image_mean, per_image_sd, at_ref, level_mass = [], [], [], []
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            grabbed.clear()
            model(imgs)
            raw = grabbed[0]
            B = raw.shape[0]
            if block.z_mode == "trilinear":
                z = block.depth(raw)                         # (B, n_z, H, W)
                zf = z.reshape(B, -1)
                # equivalent per-level mass, for a like-for-like plot vs attention
                idx = z.reshape(-1) * (L - 1)
                hist = torch.zeros(L)
                lo = idx.floor().long().clamp(0, L - 1)
                hi = (lo + 1).clamp(max=L - 1)
                frac = idx - lo.float()
                hist.index_add_(0, lo, 1 - frac)
                hist.index_add_(0, hi, frac)
                level_mass.append((hist / hist.sum()).numpy())
            else:
                w = torch.softmax(raw.view(B, -1, L, *raw.shape[-2:]), dim=2)
                level_mass.append(w.mean(dim=(0, 1, 3, 4)).numpy())
                # expected level, rescaled to the same [0,1] axis as trilinear z
                lv = torch.arange(L, dtype=torch.float32).view(1, 1, L, 1, 1)
                zf = (w * lv).sum(dim=2).reshape(B, -1) / (L - 1)
            per_image_mean.append(zf.mean(dim=1).numpy())
            per_image_sd.append(zf.std(dim=1).numpy())
            at_ref.append((zf > 0.9).float().mean(dim=1).numpy())

    handle.remove()
    pim = np.concatenate(per_image_mean)
    return {
        "tag": tag,
        "mean_z": float(pim.mean()),
        "spatial_sd": float(np.concatenate(per_image_sd).mean()),
        "image_sd": float(pim.std()),
        "at_reference": float(np.concatenate(at_ref).mean()),
        "level_mass": np.mean(level_mass, axis=0),
        "n_images": int(len(pim)),
    }


ROWS = [
    ("Oxford  trilinear", "seg_oxford_pet_pyramidal_zinit=middle_seed=0"),
    ("Oxford  trilinear +LN", "seg_oxford_pet_pyramidal_zinit=middle_lvlnorm_seed=0"),
    ("Oxford  trilinear frozen", "seg_oxford_pet_pyramidal_zinit=middle_freezez_seed=0"),
    ("Oxford  attention", "seg_oxford_pet_pyramidal_attention_zinit=middle_seed=0"),
    ("Oxford  attention +LN", "seg_oxford_pet_pyramidal_attention_zinit=middle_lvlnorm_seed=0"),
    ("Kvasir  trilinear", "seg_kvasir_pyramidal_zinit=middle_seed=0"),
    ("Kvasir  trilinear +LN", "seg_kvasir_pyramidal_zinit=middle_lvlnorm_seed=0"),
    ("Kvasir  attention", "seg_kvasir_pyramidal_attention_zinit=middle_seed=0"),
    ("Kvasir  attention +LN", "seg_kvasir_pyramidal_attention_zinit=middle_lvlnorm_seed=0"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figure", action="store_true", help="also write fig4_z_usage")
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()

    results = []
    print(f"{'config':28s} {'mean z':>7s} {'spatial sd':>11s} {'image sd':>9s} "
          f"{'>0.9':>6s}  level mass  " + "  ".join(f"{n[:4]:>5s}" for n in LEVEL_NAMES))
    for label, tag in ROWS:
        if not os.path.exists(os.path.join(CKPTS, tag + ".pth")):
            print(f"{label:28s} (no checkpoint)")
            continue
        r = z_stats(tag, args.max_batches)
        r["label"] = label
        results.append(r)
        mass = "  ".join(f"{m:5.3f}" for m in r["level_mass"])
        print(f"{label:28s} {r['mean_z']:7.3f} {r['spatial_sd']:11.3f} "
              f"{r['image_sd']:9.3f} {r['at_reference']:6.3f}              {mass}")

    if args.figure and results:
        make_figure(results)


def make_figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_figures import BLUE, ORANGE, AQUA, INK, INK2, INK3, GRID, _save, _tidy

    # Colour carries the intervention under study (level-norm on/off). Only the
    # trilinear head is plotted, matching the paper's z-usage table; the
    # attention rows are printed above for the ablation.
    keep = [r for r in results if "frozen" not in r["label"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    styles = [("trilinear", False, BLUE, ""), ("trilinear", True, ORANGE, "")]
    for ax, ds in zip(axes, ("Oxford", "Kvasir")):
        x = np.arange(len(LEVEL_NAMES))
        w = 0.8 / len(styles)
        for i, (head, ln, colour, hatch) in enumerate(styles):
            want = f"{ds}  {head}" + (" +LN" if ln else "")
            row = next((r for r in keep if r["label"] == want), None)
            if row is None:
                continue
            ax.bar(x + (i - (len(styles) - 1) / 2) * w, row["level_mass"],
                   width=w * 0.88, color=colour, hatch=hatch, edgecolor="white",
                   linewidth=0.5, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(["p1\nshallowest", "b", "d2", "d1\nreference"], fontsize=7)
        ax.set_ylabel("share of sampling weight")
        ax.set_title(ds, loc="left", color=INK)
        _tidy(ax)
        ax.grid(axis="x", visible=False)

    from matplotlib.patches import Patch
    legend = [Patch(facecolor=BLUE, label="no level-norm"),
              Patch(facecolor=ORANGE, label="with level-norm")]
    fig.legend(handles=legend, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Per-level normalisation moves sampling weight off the reference level",
                 fontsize=9, color=INK, x=0.5, y=1.03)
    fig.tight_layout()
    _save(fig, "fig4_z_usage")


if __name__ == "__main__":
    main()
