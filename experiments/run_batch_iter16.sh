#!/usr/bin/env bash
# Iteration-16 batch (~22-25 h, sequential, priority-ordered).
#
# A (~1.3 h) Classification reruns WITH checkpoints (run_cls.py --save-models
#            added 2026-08-11; no cls checkpoint was ever saved before).
#            Feeds two pending analyses: CIFAR-100 per-class redistribution
#            and the CIFAR z-usage probe for the "level-norm mildly negative
#            on CIFAR" open question (RESULTS S8).
# B (~8 h)   Level-aware offset scaling (--offset-scale-by-level): the last
#            untested top-5 block improvement. Offsets
#            were predicted in reference-level pixels regardless of z; with
#            level-norm the depth branch actually lives at coarse levels
#            (mean z .39-.68), so the reach mismatch is live in the shipped
#            config. Compare vs existing cheap-z seeds 0-4, same-seed paired.
# C (~8 h)   Depth-branch random init std 0.01 (--z-weight-init-std): cut
#            CIFAR-10 seed variance 4x at no mean cost; adopt-or-reject for
#            segmentation. Seed variance is the binding constraint of the
#            whole study (RESULTS S2 resolution floors).
# D (~8 h)   Offset-branch LR x0.1 (--offset-lr-mult, DCN practice; param
#            groups added to engine.train_segmentation 2026-08-11).
#
# All seg arms are cheap-z (the shipped config) at the ~19k bracket, 5 seeds,
# both datasets. Gate per the investigation heuristics: an arm graduates only
# with a mechanism + a plateau + held-out support.
mkdir -p results/logs

OX="--dataset oxford_pet --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
KV="--dataset kvasir --size 96 96 --batch-size 64 --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
CHEAP="--head pyramidal --level-norm --z-shared"
CLS="--epochs 150 --num-workers 6 --save-models"

run() {
    local script="$1"; local name="$2"; shift 2
    echo "=== $(date '+%H:%M:%S') starting $name ==="
    micromamba run -n ml_env python3 "experiments/${script}" "$@" \
        > "results/logs/iter16_${name}.log" 2>&1 \
        || echo "!!! $name FAILED (see results/logs/iter16_${name}.log)"
    echo "=== $(date '+%H:%M:%S') finished $name ==="
}

# --- A. cls checkpoints for the pending analyses -----------------------------
run run_cls.py c100_pyr_ckpt    --dataset cifar100 $CLS --head pyramidal --seeds 0
run run_cls.py c100_single_ckpt --dataset cifar100 $CLS --head pyramidal_single_level --seeds 0
run run_cls.py c10_pyr_ckpt     --dataset cifar10  $CLS --head pyramidal --seeds 0
run run_cls.py c10_pyr_ln_ckpt  --dataset cifar10  $CLS --head pyramidal --level-norm --seeds 0

# --- B. level-aware offset scaling -------------------------------------------
run run_seg.py kv_offscale $KV $CHEAP --offset-scale-by-level --seeds 0 1 2 3 4
run run_seg.py ox_offscale $OX $CHEAP --offset-scale-by-level --seeds 0 1 2 3 4

# --- C. depth-branch random init ---------------------------------------------
run run_seg.py kv_zwstd $KV $CHEAP --z-weight-init-std 0.01 --seeds 0 1 2 3 4
run run_seg.py ox_zwstd $OX $CHEAP --z-weight-init-std 0.01 --seeds 0 1 2 3 4

# --- D. offset-branch LR multiplier ------------------------------------------
run run_seg.py kv_offlr $KV $CHEAP --offset-lr-mult 0.1 --seeds 0 1 2 3 4
run run_seg.py ox_offlr $OX $CHEAP --offset-lr-mult 0.1 --seeds 0 1 2 3 4

echo "=== iteration-16 batch complete ==="
