#!/usr/bin/env python3
"""Generate the paper figures from results/ and kaggle_artifacts/.

    micromamba run -n ml_env python3 experiments/make_figures.py

Writes PDF and PNG (the PNGs are embedded in results/RESULTS.md) of:

  fig1_capacity_parity   -- the protocol figure: every head plotted against its
                            parameter count, so "does the pyramid beat a plain
                            wider baseline" is read off the page.
  fig2_cityscapes_class  -- the mechanism figure: per-class IoU delta vs the
                            width-matched control, classes ordered by pixel
                            frequency, both seeds.
  fig3_generalisation    -- train loss -> val loss dumbbells, showing where the
                            pyramid's extra fitting power survives to test.

Palette: slots 1-3 of the reference categorical palette, validated all-pairs in
light mode (worst CVD dE 9.2, worst normal-vision dE 24.0). Aqua sits below 3:1
on the light surface, so every aqua mark is directly labelled -- do not reuse it
for an unlabelled series.
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "axes.edgecolor": INK3,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "grid.color": GRID, "grid.linewidth": 0.6,
})


def _tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="both", zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- data loading

def _opts(row):
    try:
        return json.loads(row["options"])
    except Exception:
        return {}


def _canonical(options):
    """Drop rows carrying an intervention that was tested and rejected, so the
    capacity figure shows the design space we actually recommend rather than
    every probe ever run (residual cross-scale, DCNv2 mask, weight-decay and
    depth-init sweeps all live in RESULTS.md section 4 instead)."""
    o = json.loads(options)
    return not (o.get("use_mask") or o.get("residual_cross_scale")
                or "weight_decay" in o or "z_weight_init_std" in o
                or "offset_mode" in o or "offset_lr_mult" in o
                or o.get("scale_offsets_by_level") or o.get("offset_no_decay")
                or "epochs" in o)  # non-default schedule (CIFAR-100 300-epoch runs)


def _seg(dataset):
    d = pd.read_csv(os.path.join(ROOT, "results", "segmentation.csv"))
    d = d[(d["dataset"] == dataset) & d["options"].apply(_canonical)]
    return d


def _cls(dataset):
    d = pd.read_csv(os.path.join(ROOT, "results", "classification.csv"))
    d = d[(d["dataset"] == dataset) & d["options"].apply(_canonical)]
    # checkpoint reruns of a seed duplicate its row
    return d.drop_duplicates(["head", "params", "options", "seed"])


def _wide(name):
    """cityscapes.csv / ade20k.csv: one dataset per file, no 'dataset' column."""
    d = pd.read_csv(os.path.join(ROOT, "results", f"{name}.csv"))
    return d[d["options"].apply(_canonical)]


def _group(d, head, metric, level_norm=None, z_shared=None):
    """(params, mean, sd, n) per parameter count for one head type."""
    sub = d[d["head"] == head]
    if level_norm is not None:
        sub = sub[sub.apply(lambda r: _opts(r).get("use_level_norm", False) == level_norm, axis=1)]
    if z_shared is not None:
        # z_per_kernel_point == False is what --z-shared writes
        sub = sub[sub.apply(
            lambda r: (_opts(r).get("z_per_kernel_point", True) is False) == z_shared, axis=1)]
    if sub.empty:
        return []
    g = sub.groupby("params")[metric].agg(["mean", "std", "count"]).reset_index()
    return [(int(row["params"]), float(row["mean"]),
             0.0 if pd.isna(row["std"]) else float(row["std"]), int(row["count"]))
            for _, row in g.iterrows()]


# --------------------------------------------------------- fig 1: capacity


PANELS = [
    ("Oxford-IIIT Pet", "seg", "oxford_pet", "final_dice", "Dice", True),
    ("Kvasir-SEG", "seg", "kvasir", "final_dice", "Dice", True),
    ("CIFAR-10", "cls", "cifar10", "final_acc", "Top-1 accuracy", False),
    ("CIFAR-100", "cls", "cifar100", "final_acc", "Top-1 accuracy", False),
    ("Cityscapes", "wide", "cityscapes", "final_miou", "mIoU (dataset)", True),
    ("ADE20K", "wide", "ade20k", "final_miou", "mIoU (dataset)", True),
]


def fig_capacity():
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.6))
    for ax, (title, kind, ds, metric, ylab, lvlnorm) in zip(axes.ravel(), PANELS):
        d = {"seg": _seg, "cls": _cls, "wide": _wide}[kind](ds)

        plain = sorted(_group(d, "plain_conv", metric))
        single = sorted(_group(d, "pyramidal_single_level", metric, level_norm=False))
        single_ln = sorted(_group(d, "pyramidal_single_level", metric, level_norm=True))
        # The z-sharing axis is carried by marker fill within the trilinear hue,
        # and the normalised control by fill within the baseline hue. The
        # attention variant is reported in the paper's ablation, not here.
        pyr = sorted(_group(d, "pyramidal", metric, level_norm=lvlnorm, z_shared=False))
        cheap = sorted(_group(d, "pyramidal", metric, level_norm=lvlnorm, z_shared=True))

        # the two baselines are ladders: plot as lines so the reader can ask
        # "is the pyramid point above or below the curve at its own width?"
        for pts, colour, style, label in (
            (plain, INK3, ":", "plain conv"),
            (single, BLUE, "-", "spatial-only DCN"),
        ):
            if not pts:
                continue
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            e = [p[2] for p in pts]
            ax.errorbar(x, y, yerr=e, color=colour, linestyle=style, linewidth=1.2,
                        marker="o", markersize=3.5, capsize=2, elinewidth=0.7,
                        label=label, zorder=3)
        # normalised control: same hue, hollow
        for px, py, pe, _n in single_ln:
            ax.errorbar([px], [py], yerr=[pe], color=BLUE, marker="o", markersize=4.5,
                        markerfacecolor="white", capsize=2, elinewidth=0.7,
                        linestyle="none", zorder=4,
                        label="spatial-only DCN + level-norm"
                        if (px, "ln") == (single_ln[0][0], "ln") else None)

        for pts, colour, marker, label in (
            (cheap, ORANGE, "o", "pyramidal, one z per location"),
            (pyr, ORANGE, "s", "pyramidal, z per kernel point"),
        ):
            for px, py, pe, _n in pts:
                ax.errorbar([px], [py], yerr=[pe], color=colour, marker=marker,
                            markersize=5, capsize=2, elinewidth=0.7, linestyle="none",
                            markerfacecolor="white" if marker == "s" else colour,
                            zorder=4, label=label if (px, label) == (pts[0][0], label) else None)

        _tidy(ax)
        ax.set_title(title, color=INK, loc="left")
        ax.set_ylabel(ylab)
        ax.set_xlabel("trainable parameters")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
        ax.margins(x=0.12, y=0.14)

    handles, labels = axes[0][0].get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            h2.append(h)
            l2.append(l)
    fig.legend(h2, l2, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Accuracy against parameter count: a pyramid marker counts only if it "
                 "sits above the spatial-only ladder at its own width",
                 fontsize=9, color=INK, x=0.5, y=1.005)
    fig.tight_layout()
    _save(fig, "fig1_capacity_parity")


# ------------------------------------------------- fig 2: cityscapes per class

def fig_cityscapes_classes():
    P = os.path.join(ROOT, "kaggle_artifacts", "per_image")
    a = "cityscapes_pyramidal_attention_ch=64_zshared_lvlnorm_seed=%d_per_class.csv"
    s = "cityscapes_pyramidal_single_level_ch=66_seed=%d_per_class.csv"
    seeds = [d for d in (0, 1) if os.path.exists(os.path.join(P, a % d))
             and os.path.exists(os.path.join(P, s % d))]
    if not seeds:
        print("skipping fig2: no cityscapes per-class dumps")
        return

    ref = pd.read_csv(os.path.join(P, a % seeds[0])).set_index("name")
    order = ref["gt_pixel_frac"].sort_values().index  # rarest first (bottom-up)
    deltas = {}
    for sd in seeds:
        x = pd.read_csv(os.path.join(P, a % sd)).set_index("name")
        y = pd.read_csv(os.path.join(P, s % sd)).set_index("name")
        deltas[sd] = (x["iou"] - y["iou"]).loc[order]

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ypos = np.arange(len(order))
    h = 0.38
    for i, (sd, colour) in enumerate(zip(seeds, (BLUE, ORANGE))):
        ax.barh(ypos + (i - (len(seeds) - 1) / 2) * h, deltas[sd].to_numpy(),
                height=h * 0.92, color=colour, label=f"seed {sd}", zorder=3)
    ax.axvline(0, color=INK2, linewidth=0.8, zorder=4)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{c}  ({ref.loc[c, 'gt_pixel_frac'] * 100:.2f}%)" for c in order],
                       fontsize=6.5)
    ax.set_xlabel("IoU difference:  cross-scale attention  −  spatial-only DCN")
    _tidy(ax)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")

    from scipy.stats import spearmanr
    rhos = [spearmanr(ref["gt_pixel_frac"].loc[order], deltas[sd])[0] for sd in seeds]
    note = "  ".join(f"seed {sd}: $\\rho$={r:+.2f}" for sd, r in zip(seeds, rhos))
    ax.set_title("Cross-scale sampling trades frequent-class accuracy for rare-class accuracy\n"
                 f"classes ordered by share of labelled pixels (in brackets); "
                 f"Spearman($\\rho$) of $\\Delta$IoU vs frequency — {note}",
                 loc="left", color=INK, fontsize=8)
    fig.tight_layout()
    _save(fig, "fig2_cityscapes_per_class")


# ------------------------------------------------ fig 3: generalisation gap

def _tb_runs():
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "tensorboard_logs", "*"))):
        out.setdefault(re.sub(r"_\d{10}$", "", os.path.basename(p)), []).append(p)
    return out


def _train_val(runs, tag, seeds=range(10), tail=0.1):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tr, va = [], []
    for sd in seeds:
        key = f"{tag}_seed={sd}"
        if key not in runs:
            continue
        ea = EventAccumulator(sorted(runs[key])[-1], size_guidance={"scalars": 200000})
        ea.Reload()
        d = {t: np.array([e.value for e in ea.Scalars(t)]) for t in ea.Tags()["scalars"]}
        v = d.get("epoch_loss/train", d.get("Loss/train"))
        if v is None or not len(v):
            continue
        tr.append(v[-max(1, int(len(v) * tail)):].mean())
        if "Loss/test" in d:
            va.append(d["Loss/test"][-1])
    if not tr or not va:
        return None
    return float(np.mean(tr)), float(np.mean(va))


GAP_ROWS = [
    ("Oxford-IIIT Pet", [
        ("plain ch16", "seg_oxford_pet_plain_conv"),
        ("spatial-only+LN ch17", "seg_oxford_pet_pyramidal_single_level_zinit=middle_lvlnorm_ch=17"),
        ("trilinear+LN", "seg_oxford_pet_pyramidal_zinit=middle_lvlnorm"),
    ]),
    ("Kvasir-SEG", [
        ("plain ch16", "seg_kvasir_plain_conv"),
        ("spatial-only+LN ch17", "seg_kvasir_pyramidal_single_level_zinit=middle_lvlnorm_ch=17"),
        ("trilinear+LN", "seg_kvasir_pyramidal_zinit=middle_lvlnorm"),
    ]),
]


def fig_generalisation():
    runs = _tb_runs()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, (title, rows) in zip(axes, GAP_ROWS):
        labels, trains, vals = [], [], []
        for label, tag in rows:
            tv = _train_val(runs, tag)
            if tv is None:
                continue
            labels.append(label)
            trains.append(tv[0])
            vals.append(tv[1])
        y = np.arange(len(labels))[::-1]
        for yi, t, v in zip(y, trains, vals):
            ax.plot([t, v], [yi, yi], color=INK3, linewidth=1.0, zorder=2)
        ax.scatter(trains, y, color=BLUE, s=26, zorder=3, label="train loss")
        ax.scatter(vals, y, color=ORANGE, s=26, zorder=3, label="val loss")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        _tidy(ax)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("loss (BCE + Dice)")
    axes[0].legend(loc="lower right")
    fig.suptitle("Where the pyramid's extra fitting power survives to validation",
                 fontsize=9, color=INK, x=0.5, y=1.02)
    fig.tight_layout()
    _save(fig, "fig3_generalisation_gap")


# ------------------------------------------ fig 5: fitting power at parity

def _tails(runs, tag, seeds=range(10), tail=0.1):
    """Per-seed tail-mean train loss for one tag."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    out = {}
    for sd in seeds:
        key = f"{tag}_seed={sd}"
        if key not in runs:
            continue
        ea = EventAccumulator(sorted(runs[key])[-1], size_guidance={"scalars": 200000})
        ea.Reload()
        d = {t: np.array([e.value for e in ea.Scalars(t)]) for t in ea.Tags()["scalars"]}
        v = d.get("epoch_loss/train", d.get("Loss/train"))
        if v is not None and len(v):
            out[sd] = float(v[-max(1, int(len(v) * tail)):].mean())
    return out


# (row label, arm tag, control tag) -- every capacity-parity fitting comparison
# of the trilinear family in RESULTS §6. The attention variant's comparisons
# belong to the paper's ablation section, not to this figure.
FIT_ROWS = [
    ("Oxford Pet · full-$z$", "seg_oxford_pet_pyramidal_zinit=middle_lvlnorm",
     "seg_oxford_pet_pyramidal_single_level_zinit=middle_lvlnorm_ch=17"),
    ("Oxford Pet · cheap-$z$", "seg_oxford_pet_pyramidal_zinit=middle_zshared_lvlnorm",
     "seg_oxford_pet_pyramidal_single_level_zinit=middle_lvlnorm_ch=17"),
    ("Kvasir-SEG · cheap-$z$", "seg_kvasir_pyramidal_zinit=middle_zshared_lvlnorm",
     "seg_kvasir_pyramidal_single_level_zinit=middle_lvlnorm_ch=17"),
    ("CIFAR-10 · full-$z$", "cls_cifar10_pyramidal",
     "cls_cifar10_pyramidal_single_level_ch=18"),
    ("CIFAR-100 · full-$z$", "cls_cifar100_pyramidal",
     "cls_cifar100_pyramidal_single_level_ch=17"),
    ("Cityscapes · cheap-$z$", "cityscapes_pyramidal_ch=64_zshared_lvlnorm",
     "cityscapes_pyramidal_single_level_ch=66_lvlnorm"),
    ("ADE20K · cheap-$z$", "ade20k_pyramidal_ch=64_zshared_lvlnorm",
     "ade20k_pyramidal_single_level_ch=66_lvlnorm"),
]


def fig_fitting():
    runs = _tb_runs()
    # Kaggle runs, plus the Cityscapes normalised-control seeds trained locally
    for p in sorted(glob.glob(os.path.join(ROOT, "kaggle_artifacts", "tb_logs", "*"))
                    + glob.glob(os.path.join(ROOT, "results", "cityscapes_local", "tb", "*"))):
        runs.setdefault(re.sub(r"_\d{10}$", "", os.path.basename(p)), []).append(p)

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    labels, y = [], 0
    positions = []
    for label, tag_a, tag_b in FIT_ROWS:
        ta, tb = _tails(runs, tag_a), _tails(runs, tag_b)
        common = sorted(set(ta) & set(tb))
        if not common:
            continue
        rel = [(ta[s] - tb[s]) / tb[s] * 100.0 for s in common]
        ax.scatter(rel, [y] * len(rel), s=12, color=ORANGE, alpha=0.45, zorder=3,
                   edgecolors="none")
        ax.scatter([np.mean(rel)], [y], s=48, color=ORANGE, zorder=4,
                   edgecolors="white", linewidths=0.8)
        labels.append(label)
        positions.append(y)
        y -= 1

    ax.axvline(0, color=INK3, linewidth=0.8, linestyle="--", zorder=2)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=7.5)
    _tidy(ax)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(positions[-1] - 0.6, 1.1)
    ax.set_xlabel("training-loss change at parameter parity vs the "
                  "width-matched control (%) — one dot per seed pair")
    ax.annotate("fits better  $\\leftarrow$", xy=(-0.6, 0.72), ha="right",
                color=INK2, fontsize=7.5, annotation_clip=False)
    ax.annotate("$\\rightarrow$  fits worse", xy=(0.6, 0.72), ha="left",
                color=INK2, fontsize=7.5, annotation_clip=False)
    ax.set_title("Training loss against a control holding equal-or-more parameters\n"
                 "(normalised control for every dense-prediction row)",
                 loc="left", color=INK, fontsize=8.5)
    fig.tight_layout()
    _save(fig, "fig5_fitting_advantage")


def _save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"{stem}.{ext}")
        fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", os.path.join(OUT, stem) + ".{pdf,png}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["capacity", "cityscapes", "gap", "fitting"]
    if "capacity" in which:
        fig_capacity()
    if "cityscapes" in which:
        fig_cityscapes_classes()
    if "gap" in which:
        fig_generalisation()
    if "fitting" in which:
        fig_fitting()
