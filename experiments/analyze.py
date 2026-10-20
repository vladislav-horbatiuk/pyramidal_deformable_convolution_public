#!/usr/bin/env python3
"""Regenerate the result tables from the artifacts in results/ and
tensorboard_logs/ (the latter is not part of this repository; see README.md). No training, no GPU; safe to rerun any time.

    micromamba run -n ml_env python3 experiments/analyze.py [section ...]

Sections: `agg` (mean±std per config), `parity` (paired per-image Wilcoxon),
`cifar` (Welch t-tests over seeds), `loss` (train loss / val loss / gap),
`loss-sig` (seed-level significance of the train-loss differences),
`cityscapes` (Kaggle runs). Default: all of them.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PER_IMAGE = os.path.join(ROOT, "results", "per_image")
TB = os.path.join(ROOT, "tensorboard_logs")
KAGGLE_TB = os.path.join(ROOT, "kaggle_artifacts", "tb_logs")
# Cityscapes normalised-control seeds 1-2 were trained locally
CITY_LOCAL_TB = os.path.join(ROOT, "results", "cityscapes_local", "tb")


def config_label(row):
    """Human-readable config from the head name + options JSON. `params` is
    what actually distinguishes widths, since --channels is not in the CSV."""
    opts = json.loads(row["options"]) if isinstance(row["options"], str) else {}
    bits = [row["head"]]
    for key, name in (("use_level_norm", "lvlnorm"), ("z_per_kernel_point", None),
                      ("freeze_z", "freezez"), ("use_mask", "mask")):
        if key == "z_per_kernel_point":
            if opts.get(key) is False:
                bits.append("zshared")
        elif opts.get(key):
            bits.append(name)
    if "weight_decay" in opts:
        bits.append(f"wd={opts['weight_decay']:g}")
    return "+".join(bits)


# --------------------------------------------------------------------------
# aggregate tables
# --------------------------------------------------------------------------

def section_agg():
    seg = pd.read_csv(os.path.join(ROOT, "results", "segmentation.csv"))
    cls = pd.read_csv(os.path.join(ROOT, "results", "classification.csv"))
    seg["cfg"] = seg.apply(config_label, axis=1)
    cls["cfg"] = cls.apply(config_label, axis=1)

    for ds in ("oxford_pet", "kvasir"):
        sub = seg[seg["dataset"] == ds]
        if sub.empty:
            continue
        g = sub.groupby(["cfg", "params"]).agg(
            n=("seed", "count"),
            dice=("final_dice", "mean"), dice_sd=("final_dice", "std"),
            bf1=("final_bf1", "mean"),
        ).reset_index().sort_values("dice", ascending=False)
        print(f"\n===== {ds}: final micro Dice =====")
        print(g.to_string(index=False))

    for ds in ("cifar10", "cifar100"):
        sub = cls[cls["dataset"] == ds]
        if sub.empty:
            continue
        g = sub.groupby(["cfg", "params"]).agg(
            n=("seed", "count"),
            acc=("final_acc", "mean"), acc_sd=("final_acc", "std"),
        ).reset_index().sort_values("acc", ascending=False)
        print(f"\n===== {ds}: final top-1 =====")
        print(g.to_string(index=False))


# --------------------------------------------------------------------------
# paired per-image Wilcoxon
# --------------------------------------------------------------------------

def _per_image(tag, seeds):
    cols = []
    for s in seeds:
        path = os.path.join(PER_IMAGE, f"{tag}_seed={s}.csv")
        if os.path.exists(path):
            cols.append(pd.read_csv(path).sort_values("image_idx")["dice"].to_numpy())
    if not cols:
        raise FileNotFoundError(f"no per-image dumps for {tag}")
    return np.mean(np.stack(cols), axis=0), len(cols)


def _compare(name, tag_a, tag_b, seeds_a=range(5), seeds_b=range(5)):
    a, na = _per_image(tag_a, seeds_a)
    b, nb = _per_image(tag_b, seeds_b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    _, p = wilcoxon(a, b)
    d = a - b
    print(f"{name:52s} A={a.mean():.4f}(n{na}) B={b.mean():.4f}(n{nb}) "
          f"D={d.mean():+.4f} p={p:.2e} imgs={n} A-wins={100 * (d > 0).mean():.0f}%")


OX = "seg_oxford_pet_"
KV = "seg_kvasir_"

PARITY = [
    ("OXFORD: capacity parity", [
        ("attn+ln(23335) vs single ch19(23465)",
         OX + "pyramidal_attention_zinit=middle_lvlnorm",
         OX + "pyramidal_single_level_zinit=middle_ch=19", range(5), range(5)),
        ("pyr+ln(19420) vs single ch17(19161)",
         OX + "pyramidal_zinit=middle_lvlnorm",
         OX + "pyramidal_single_level_zinit=middle_ch=17", range(5), range(5)),
    ]),
    ("OXFORD: level-norm ablation", [
        ("pyr+ln vs pyr", OX + "pyramidal_zinit=middle_lvlnorm",
         OX + "pyramidal_zinit=middle", range(5), range(5)),
        ("attn+ln vs attn", OX + "pyramidal_attention_zinit=middle_lvlnorm",
         OX + "pyramidal_attention_zinit=middle", range(5), range(5)),
    ]),
    ("KVASIR: capacity parity", [
        ("attn+ln(23335) vs single ch19(23465)",
         KV + "pyramidal_attention_zinit=middle_lvlnorm",
         KV + "pyramidal_single_level_zinit=middle_ch=19", range(5), range(3)),
        ("pyr+ln(19420) vs single ch17(19161)",
         KV + "pyramidal_zinit=middle_lvlnorm",
         KV + "pyramidal_single_level_zinit=middle_ch=17", range(5), range(5)),
    ]),
    ("KVASIR: level-norm ablation and vs plain", [
        ("pyr+ln vs pyr", KV + "pyramidal_zinit=middle_lvlnorm",
         KV + "pyramidal_zinit=middle", range(5), range(3)),
        ("attn+ln vs attn", KV + "pyramidal_attention_zinit=middle_lvlnorm",
         KV + "pyramidal_attention_zinit=middle", range(5), range(3)),
        ("single ch16 vs plain ch16 (spatial deform)",
         KV + "pyramidal_single_level_zinit=middle",
         KV + "plain_conv", range(5), range(5)),
    ]),
]


def section_parity():
    for group, rows in PARITY:
        print(f"\n--- {group} ---")
        for name, a, b, sa, sb in rows:
            try:
                _compare(name, a, b, sa, sb)
            except FileNotFoundError as e:
                print(f"{name:52s} skipped ({e})")


# --------------------------------------------------------------------------
# CIFAR: Welch t-tests over seeds
# --------------------------------------------------------------------------

CIFAR_PAIRS = [
    ("cifar10", "pyramidal", 82581, "pyramidal_single_level", 88552),
    ("cifar10", "pyramidal_attention", 86496, "pyramidal_single_level", 88552),
    ("cifar10", "pyramidal", 82581, "plain_conv", 77850),
    ("cifar100", "pyramidal", 95991, "pyramidal_single_level", 93870),
    ("cifar100", "pyramidal_attention", 99906, "pyramidal_single_level", 93870),
    # width-matched controls (ch16 holds fewer params than either pyramid head)
    ("cifar100", "pyramidal", 95991, "pyramidal_single_level", 97880),
    ("cifar100", "pyramidal_attention", 99906, "pyramidal_single_level", 101962),
]


def section_cifar():
    d = pd.read_csv(os.path.join(ROOT, "results", "classification.csv"))
    print("\n--- CIFAR: Welch t-test over seeds ---")
    for ds, ha, pa, hb, pb in CIFAR_PAIRS:
        def sel(h, p):
            m = (d["dataset"] == ds) & (d["head"] == h) & (d["params"] == p)
            # only the default-regularisation, default-schedule runs
            m &= ~d["options"].str.contains("weight_decay|epochs", na=False)
            # one row per seed: checkpoint reruns of a seed duplicate its row
            return d[m].drop_duplicates("seed")["final_acc"].to_numpy()
        a, b = sel(ha, pa), sel(hb, pb)
        if len(a) < 2 or len(b) < 2:
            print(f"{ds} {ha}({pa}) vs {hb}({pb}): too few seeds "
                  f"({len(a)}/{len(b)})")
            continue
        _, p = ttest_ind(a, b, equal_var=False)
        print(f"{ds:9s} {ha}({pa}) n={len(a)} {a.mean():.4f}±{a.std(ddof=1):.4f} "
              f"vs {hb}({pb}) n={len(b)} {b.mean():.4f}±{b.std(ddof=1):.4f} "
              f"D={a.mean() - b.mean():+.4f} p={p:.3f}")


# --------------------------------------------------------------------------
# train loss / val loss / generalization gap
# --------------------------------------------------------------------------

def _tb_runs(root):
    out = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(root, "*"))):
        out[re.sub(r"_\d{10}$", "", os.path.basename(path))].append(path)
    return out


def _scalars(run_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 200000})
    ea.Reload()
    return {t: np.array([e.value for e in ea.Scalars(t)]) for t in ea.Tags()["scalars"]}


def _loss_stats(runs, tag, seeds, tail=0.1):
    tr, va = [], []
    for s in seeds:
        key = f"{tag}_seed={s}"
        if key not in runs:
            continue
        d = _scalars(sorted(runs[key])[-1])
        v = d.get("epoch_loss/train", d.get("Loss/train"))
        if v is None or len(v) == 0:
            continue
        tr.append(v[-max(1, int(len(v) * tail)):].mean())
        if "Loss/test" in d:
            va.append(d["Loss/test"][-1])
    if not tr:
        return None
    # train and val seed counts differ: Loss/test was only added to the
    # classification loop in iteration 6, so earlier cls runs have no val curve.
    # Reporting one n for both would overstate the val numbers.
    return np.mean(tr), np.std(tr), (np.mean(va) if va else None), len(tr), len(va)


LOSS_GROUPS = {
    "OXFORD PET (60 ep)": [
        ("plain ch16 14561", OX + "plain_conv"),
        ("single ch17 19161", OX + "pyramidal_single_level_zinit=middle_ch=17"),
        ("single+ln ch17 19195", OX + "pyramidal_single_level_zinit=middle_lvlnorm_ch=17"),
        ("pyramidal+ln 19420", OX + "pyramidal_zinit=middle_lvlnorm"),
        ("single ch19 23465", OX + "pyramidal_single_level_zinit=middle_ch=19"),
        ("attention+ln 23335", OX + "pyramidal_attention_zinit=middle_lvlnorm"),
    ],
    "KVASIR-SEG (60 ep)": [
        ("plain ch16 14561", KV + "plain_conv"),
        ("single ch17 19161", KV + "pyramidal_single_level_zinit=middle_ch=17"),
        ("single+ln ch17 19195", KV + "pyramidal_single_level_zinit=middle_lvlnorm_ch=17"),
        ("pyramidal+ln 19420", KV + "pyramidal_zinit=middle_lvlnorm"),
        ("single ch19 23465", KV + "pyramidal_single_level_zinit=middle_ch=19"),
        ("attention+ln 23335", KV + "pyramidal_attention_zinit=middle_lvlnorm"),
    ],
    "CIFAR-10 (150 ep)": [
        ("plain 77850", "cls_cifar10_plain_conv"),
        ("single ch18 88552", "cls_cifar10_pyramidal_single_level_ch=18"),
        ("pyramidal 82581", "cls_cifar10_pyramidal"),
        ("attention 86496", "cls_cifar10_pyramidal_attention"),
    ],
    "CIFAR-100 (150 ep)": [
        ("plain 91260", "cls_cifar100_plain_conv"),
        ("single 93870", "cls_cifar100_pyramidal_single_level"),
        ("single ch17 97880", "cls_cifar100_pyramidal_single_level_ch=17"),
        ("pyramidal 95991", "cls_cifar100_pyramidal"),
        ("attention 99906", "cls_cifar100_pyramidal_attention"),
    ],
}


def section_loss(seeds=range(10)):
    runs = _tb_runs(TB)
    for group, rows in LOSS_GROUPS.items():
        print(f"\n===== {group}: train loss (last 10% of steps) vs val loss =====")
        print(f"  {'config':22s} {'train (n)':>18s} {'val (n)':>14s} {'gap':>8s}")
        for label, tag in rows:
            st = _loss_stats(runs, tag, seeds)
            if st is None:
                print(f"  {label:22s} -- (no tensorboard runs)")
                continue
            m, sd, v, n_tr, n_va = st
            val = f"{v:.4f} ({n_va})" if v is not None else "n/a"
            gap = f"{v - m:+.4f}" if v is not None else "n/a"
            print(f"  {label:22s} {m:.4f} ± {sd:.4f} ({n_tr:d}) {val:>14s} {gap:>8s}")


# --------------------------------------------------------------------------
# Train-loss seed-level significance (the "stronger fitter" claim)
# --------------------------------------------------------------------------

def _train_curves(runs, tag, seeds=range(10)):
    """Per-seed train-loss curve (epoch_loss/train, else per-step Loss/train)."""
    out = {}
    for s in seeds:
        key = f"{tag}_seed={s}"
        if key not in runs:
            continue
        d = _scalars(sorted(runs[key])[-1])
        v = d.get("epoch_loss/train", d.get("Loss/train"))
        if v is not None and len(v):
            out[s] = v
    return out


LOSS_SIG = [
    ("OXFORD", [
        ("full-z+LN 19420 vs single ch17 19161",
         OX + "pyramidal_zinit=middle_lvlnorm",
         OX + "pyramidal_single_level_zinit=middle_ch=17"),
        ("full-z+LN 19420 vs single+LN ch17 19195",
         OX + "pyramidal_zinit=middle_lvlnorm",
         OX + "pyramidal_single_level_zinit=middle_lvlnorm_ch=17"),
        ("cheap-z 18260 vs single+LN ch17 19195",
         OX + "pyramidal_zinit=middle_zshared_lvlnorm",
         OX + "pyramidal_single_level_zinit=middle_lvlnorm_ch=17"),
        ("attention+LN 23335 vs single+LN ch19 23503",
         OX + "pyramidal_attention_zinit=middle_lvlnorm",
         OX + "pyramidal_single_level_zinit=middle_lvlnorm_ch=19"),
    ]),
    ("KVASIR", [
        ("full-z+LN 19420 vs single ch17 19161",
         KV + "pyramidal_zinit=middle_lvlnorm",
         KV + "pyramidal_single_level_zinit=middle_ch=17"),
        ("cheap-z 18260 vs single+LN ch17 19195",
         KV + "pyramidal_zinit=middle_zshared_lvlnorm",
         KV + "pyramidal_single_level_zinit=middle_lvlnorm_ch=17"),
        ("cheap-z 18260 vs single ch17 19161",
         KV + "pyramidal_zinit=middle_zshared_lvlnorm",
         KV + "pyramidal_single_level_zinit=middle_ch=17"),
        ("attention+LN 23335 vs single ch19 23465",
         KV + "pyramidal_attention_zinit=middle_lvlnorm",
         KV + "pyramidal_single_level_zinit=middle_ch=19"),
    ]),
    ("CIFAR", [
        ("C10 pyramidal 82581 vs single ch18 88552",
         "cls_cifar10_pyramidal", "cls_cifar10_pyramidal_single_level_ch=18"),
        ("C100 pyramidal 95991 vs single 93870",
         "cls_cifar100_pyramidal", "cls_cifar100_pyramidal_single_level"),
        ("C100 pyramidal 95991 vs single ch17 97880",
         "cls_cifar100_pyramidal", "cls_cifar100_pyramidal_single_level_ch=17"),
        ("C100 attention 99906 vs single ch18 101962",
         "cls_cifar100_pyramidal_attention", "cls_cifar100_pyramidal_single_level_ch=18"),
    ]),
    ("WIDE (Kaggle tb_logs)", [
        ("Cityscapes cheap-z 258470 vs single ch66 259681",
         "cityscapes_pyramidal_ch=64_zshared_lvlnorm",
         "cityscapes_pyramidal_single_level_ch=66"),
        ("ADE20K cheap-z 334057 vs single ch66 337626",
         "ade20k_pyramidal_ch=64_zshared_lvlnorm",
         "ade20k_pyramidal_single_level_ch=66"),
        # normalisation-matched controls (the paper's dense-prediction rows)
        ("Cityscapes cheap-z 258470 vs single+LN ch66 259813",
         "cityscapes_pyramidal_ch=64_zshared_lvlnorm",
         "cityscapes_pyramidal_single_level_ch=66_lvlnorm"),
        ("ADE20K cheap-z 334057 vs single+LN ch66 337758",
         "ade20k_pyramidal_ch=64_zshared_lvlnorm",
         "ade20k_pyramidal_single_level_ch=66_lvlnorm"),
    ]),
]


def section_loss_sig():
    """Does the pyramid fit the training data better, reproducibly?

    Per seed: mean train loss over the last 10 % of logged points. Welch t
    across seeds; paired t over common seeds (same seed = same data order, so
    pairing removes shared shuffle noise). Two trajectory diagnostics:
    `lead` = fraction of the training run where the arm's seed-mean curve is
    below the control's (a fit advantage held throughout, not a final-epoch
    artifact), and `slack` = relative loss decrease over the last 20 % of
    epochs per head (near 0 = converged; large = the tail mean understates
    where longer training would land).
    """
    from scipy.stats import ttest_rel
    runs = defaultdict(list)
    for root in (TB, KAGGLE_TB, CITY_LOCAL_TB):
        for key, paths in _tb_runs(root).items():
            runs[key].extend(paths)
    print("\n===== Train loss at capacity parity: seed-level significance =====")
    for group, rows in LOSS_SIG:
        print(f"\n--- {group} ---")
        for name, tag_a, tag_b in rows:
            ca, cb = _train_curves(runs, tag_a), _train_curves(runs, tag_b)
            if len(ca) < 2 or len(cb) < 2:
                print(f"  {name:46s} too few runs ({len(ca)}/{len(cb)})")
                continue
            ta = {s: v[-max(1, int(len(v) * 0.1)):].mean() for s, v in ca.items()}
            tb_ = {s: v[-max(1, int(len(v) * 0.1)):].mean() for s, v in cb.items()}
            a, b = np.array(list(ta.values())), np.array(list(tb_.values()))
            _, p_welch = ttest_ind(a, b, equal_var=False)
            common = sorted(set(ta) & set(tb_))
            p_pair = (ttest_rel([ta[s] for s in common], [tb_[s] for s in common])[1]
                      if len(common) > 1 else float("nan"))

            def mean_curve(curves, n=200):
                return np.mean([v[np.linspace(0, len(v) - 1, n).astype(int)]
                                for v in curves.values()], axis=0)

            def slack(curves):
                return np.mean([(v[-max(1, int(len(v) * 0.2))] - v[-1])
                                / max(v[-1], 1e-9) for v in curves.values()])

            lead = float((mean_curve(ca) < mean_curve(cb)).mean())
            print(f"  {name:46s} D={a.mean() - b.mean():+.4f} "
                  f"n={len(a)}/{len(b)}  welch_p={p_welch:.3f}  "
                  f"paired_p={p_pair:.1e}(n={len(common)})  lead={lead:.2f}  "
                  f"slack={slack(ca) * 100:+.1f}%/{slack(cb) * 100:+.1f}%")


# --------------------------------------------------------------------------
# Cityscapes (Kaggle)
# --------------------------------------------------------------------------

def _cityscapes_dumps():
    """Group kaggle_artifacts/per_image/*.csv by config, collecting seeds.

    Returns {config_without_seed: {"seeds": [...], "per_image": [df, ...],
    "per_class": [df indexed by class name, ...]}}."""
    out = defaultdict(lambda: {"seeds": [], "per_image": [], "per_class": []})
    # Kaggle-produced dumps and locally produced ones live side by side; the tag
    # scheme is identical, so a config run in both places just gains seeds.
    roots = [os.path.join(ROOT, "kaggle_artifacts", "per_image"),
             os.path.join(ROOT, "results", "cityscapes_local", "per_image")]
    paths = sorted(p for r in roots for p in glob.glob(os.path.join(r, "*.csv")))
    for path in paths:
        stem = os.path.basename(path)[:-len(".csv")]
        if not stem.startswith("cityscapes_"):
            continue  # ade20k_* dumps have their own analysis; don't mix
        kind = "per_class" if stem.endswith("_per_class") else "per_image"
        if kind == "per_class":
            stem = stem[:-len("_per_class")]
        m = re.search(r"_seed=(\d+)$", stem)
        if not m:
            continue
        cfg = stem[:m.start()].replace("cityscapes_", "")
        seed = int(m.group(1))
        df = pd.read_csv(path)
        df = df.set_index("name") if kind == "per_class" else df.sort_values("image_idx")
        entry = out[cfg]
        if seed not in entry["seeds"]:
            entry["seeds"].append(seed)
        entry[kind].append(df)
    return {k: v for k, v in out.items() if v["per_image"] and v["per_class"]}


def section_cityscapes():
    if not os.path.isdir(KAGGLE_TB):
        print("\n(no kaggle_artifacts/tb_logs)")
        return
    print("\n===== CITYSCAPES (Kaggle, crop 256x256, batch 4, 60 ep) =====")
    print(f"  {'run':46s} {'train':>7s} {'val':>7s} {'gap':>7s} {'mIoU':>7s} {'pixAcc':>7s}")
    tb_dirs = sorted(glob.glob(os.path.join(KAGGLE_TB, "*"))
                     + glob.glob(os.path.join(ROOT, "results", "cityscapes_local", "tb", "*")))
    for path in tb_dirs:
        d = _scalars(path)
        v = d["Loss/train"]
        tr = v[-max(1, int(len(v) * 0.1)):].mean()
        val = d["Loss/test"][-1]
        # some dirs carry the launch timestamp suffix, some don't
        name = re.sub(r"_\d{10}$", "", os.path.basename(path))
        name = name.replace("cityscapes_", "")
        print(f"  {name:46s} {tr:7.4f} {val:7.4f} {val - tr:+7.4f} "
              f"{d['mIoU/test'][-1]:7.4f} {d['PixelAcc/test'][-1]:7.4f}")

    dumps = _cityscapes_dumps()
    if not dumps:
        print("  (per-class / per-image dumps not present yet — see the "
              "--eval-only pass in experiments/KAGGLE.md)")
        return
    # canonical control = the un-normalised width-matched DCN; the +lvlnorm
    # variant (added in plan-session 3) is an *arm* here, not the reference.
    base = (next((c for c in dumps if "single_level" in c and "lvlnorm" not in c), None)
            or next((c for c in dumps if "single_level" in c), None))
    if base is None:
        print("  (no width-matched single_level dump to compare against)")
        return

    freq = dumps[base]["per_class"][0]["gt_pixel_frac"]
    order = freq.sort_values(ascending=False).index

    print("\n  per-class IoU (mean over seeds), classes by pixel frequency:")
    table = pd.DataFrame({"gt_pixel_frac": freq})
    for cfg, d in dumps.items():
        table[cfg] = pd.concat([x["iou"] for x in d["per_class"]], axis=1).mean(axis=1)
    print(table.loc[order].to_string())

    # Dataset-level mIoU weights all 19 classes equally, so the 11 rare ones
    # (<1% of pixels) drive it, while pixel accuracy and per-image mIoU are
    # driven by the frequent ones. That is why the two can disagree in sign on
    # the very same checkpoint — quantify it rather than asserting it.
    from scipy.stats import spearmanr
    is_freq = freq >= 0.008
    print(f"\n  per-class IoU delta vs {base}, ranked against class frequency:")
    for cfg, d in dumps.items():
        if cfg == base:
            continue
        deltas = []
        for seed, pc in zip(d["seeds"], d["per_class"]):
            if seed not in dumps[base]["seeds"]:
                continue
            ref = dumps[base]["per_class"][dumps[base]["seeds"].index(seed)]
            delta = pc["iou"] - ref["iou"]
            deltas.append(delta)
            rho, p = spearmanr(freq, delta)
            print(f"    {cfg:34s} seed {seed}  rho={rho:+.3f} p={p:.4f} "
                  f"improved={int((delta > 0).sum())}/{len(delta)} "
                  f"frequent={delta[is_freq].mean():+.4f} "
                  f"rare={delta[~is_freq].mean():+.4f}")
        if len(deltas) > 1:
            # Does the *pattern* replicate even when the aggregate does not?
            rho, p = spearmanr(deltas[0], deltas[1])
            agree = int((np.sign(deltas[0]) == np.sign(deltas[1])).sum())
            print(f"    {cfg:34s} seed-to-seed agreement of the per-class "
                  f"deltas: rho={rho:+.3f} p={p:.4f}, signs {agree}/{len(deltas[0])}")

    # paired per-image tests (here the sample is images, not classes)
    n_img = len(dumps[base]["per_image"][0])
    for metric in ("miou", "pixel_acc"):
        print(f"\n  paired per-image {metric} vs {base} ({n_img} val images):")
        for cfg, d in dumps.items():
            if cfg == base:
                continue
            for seed, pi in zip(d["seeds"], d["per_image"]):
                if seed not in dumps[base]["seeds"]:
                    continue
                ref = dumps[base]["per_image"][dumps[base]["seeds"].index(seed)]
                a, b = pi[metric].to_numpy(), ref[metric].to_numpy()
                n = min(len(a), len(b))
                _, p = wilcoxon(a[:n], b[:n])
                print(f"    {cfg:34s} seed {seed}  {a[:n].mean():.4f} vs "
                      f"{b[:n].mean():.4f} D={a[:n].mean() - b[:n].mean():+.4f} "
                      f"p={p:.2e}")



# --------------------------------------------------------------------------
# seed-level significance (the reproducibility question)
# --------------------------------------------------------------------------

def _per_seed_macro(tag, seeds=range(10)):
    """Macro Dice per seed: mean over test images within each run.

    The paired Wilcoxon in `parity` averages per-image scores across seeds and
    then tests across images, which answers "do these two averaged models
    differ on this test set". It cannot answer "does the difference reproduce
    across training runs" -- with few seeds it will happily call seed noise
    significant, because the seed-averaged curve it tests is treated as fixed.
    Aggregating the other way round -- image-mean first, then test across seeds
    -- answers the reproducibility question, at the cost of much less power.
    Report both; they disagree exactly when the effect is below seed noise.
    """
    out = []
    for s in seeds:
        path = os.path.join(PER_IMAGE, f"{tag}_seed={s}.csv")
        if os.path.exists(path):
            out.append(pd.read_csv(path)["dice"].mean())
    return np.array(out)


SEED_LEVEL = [
    ("KVASIR", [
        ("cheap-z 18260 vs single+LN ch17 19195",
         KV + "pyramidal_zinit=middle_zshared_lvlnorm",
         KV + "pyramidal_single_level_zinit=middle_lvlnorm_ch=17"),
        ("cheap-z 18260 vs single ch17 19161",
         KV + "pyramidal_zinit=middle_zshared_lvlnorm",
         KV + "pyramidal_single_level_zinit=middle_ch=17"),
        ("cheap-z vs full-z 19420",
         KV + "pyramidal_zinit=middle_zshared_lvlnorm",
         KV + "pyramidal_zinit=middle_lvlnorm"),
        ("attention+LN 23335 vs single ch19 23465",
         KV + "pyramidal_attention_zinit=middle_lvlnorm",
         KV + "pyramidal_single_level_zinit=middle_ch=19"),
        ("single ch16 vs plain ch16  [large effect, control]",
         KV + "pyramidal_single_level_zinit=middle", KV + "plain_conv"),
        ("level-norm on attention    [large effect, control]",
         KV + "pyramidal_attention_zinit=middle_lvlnorm",
         KV + "pyramidal_attention_zinit=middle"),
    ]),
    ("OXFORD", [
        ("cheap-z 18260 vs single+LN ch17 19195",
         OX + "pyramidal_zinit=middle_zshared_lvlnorm",
         OX + "pyramidal_single_level_zinit=middle_lvlnorm_ch=17"),
        ("full-z+LN 19420 vs single+LN ch17 19195",
         OX + "pyramidal_zinit=middle_lvlnorm",
         OX + "pyramidal_single_level_zinit=middle_lvlnorm_ch=17"),
        ("attention+LN 23335 vs single+LN ch19 23503",
         OX + "pyramidal_attention_zinit=middle_lvlnorm",
         OX + "pyramidal_single_level_zinit=middle_lvlnorm_ch=19"),
        ("level-norm on trilinear    [large effect, control]",
         OX + "pyramidal_zinit=middle_lvlnorm", OX + "pyramidal_zinit=middle"),
        ("single ch16 vs plain ch16  [large effect, control]",
         OX + "pyramidal_single_level_zinit=middle", OX + "plain_conv"),
    ]),
]


def section_seeds():
    from scipy.stats import ttest_ind
    print("\n===== Seed-level re-analysis: does the difference reproduce "
          "across training runs? =====")
    print("  Wilcoxon p = across images (seeds averaged first); "
          "Welch p = across seeds (images averaged first).")
    for group, rows in SEED_LEVEL:
        print(f"\n--- {group} ---")
        for name, ta, tb in rows:
            a, b = _per_seed_macro(ta), _per_seed_macro(tb)
            if len(a) < 2 or len(b) < 2:
                print(f"  {name:46s} too few seeds ({len(a)}/{len(b)})")
                continue
            try:
                wa, _ = _per_image(ta, range(10))
                wb, _ = _per_image(tb, range(10))
                n = min(len(wa), len(wb))
                _, pw = wilcoxon(wa[:n], wb[:n])
            except Exception:
                pw = float("nan")
            _, pt = ttest_ind(a, b, equal_var=False)
            d = a.mean() - b.mean()
            # smallest difference this seed budget could resolve at ~2 sigma
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            flag = "AGREE" if (pw < 0.05) == (pt < 0.05) else "**DISAGREE**"
            print(f"  {name:46s} D={d:+.4f} n={len(a)}/{len(b)}  "
                  f"wilcoxon_p={pw:.1e}  welch_p={pt:.3f}  "
                  f"resolvable>={2 * se:.4f}  {flag}")


SECTIONS = {
    "agg": section_agg, "parity": section_parity, "cifar": section_cifar,
    "loss": section_loss, "loss-sig": section_loss_sig,
    "cityscapes": section_cityscapes, "seeds": section_seeds,
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(SECTIONS)
    for name in wanted:
        if name not in SECTIONS:
            sys.exit(f"unknown section {name!r}; pick from {list(SECTIONS)}")
        SECTIONS[name]()
