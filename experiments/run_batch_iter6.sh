#!/usr/bin/env bash
# Iteration-6 batch (~13 h CPU, sequential). Three questions, in priority order:
#
#  A. Is the pyramid's capacity-parity loss on Kvasir a *variance* problem?
#     Iteration-5 train/val curves show the pyramidal heads reach the same or a
#     better train loss than the width-matched spatial-only control while
#     ending with a clearly larger train->val gap (Kvasir: attention+lvlnorm
#     .3218/.3997 vs single ch19 .3055/.3677). If so, more weight decay should
#     help the pyramid more than the control -- an interaction, not a shift.
#  B. Is the Kvasir scaling deficit systematic, or only at the 23k bracket?
#     Adds a third parity bracket (~21.3-21.6k).
#  C. Bring the strongest positive result (CIFAR-100) to 5 seeds, and give the
#     CIFAR-10 parity comparison enough seeds to be testable (10 vs 10).
mkdir -p results/logs

OX="--dataset oxford_pet --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
KV="--dataset kvasir --size 96 96 --batch-size 64 --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
C10="--dataset cifar10 --epochs 150 --num-workers 6"
C100="--dataset cifar100 --epochs 150 --num-workers 6"

run() {
    local script="$1" name="$2"; shift 2
    echo "=== $(date '+%H:%M:%S') starting $name ==="
    micromamba run -n ml_env python3 "experiments/${script}" "$@" \
        > "results/logs/iter6_${name}.log" 2>&1 \
        || echo "!!! $name FAILED (see results/logs/iter6_${name}.log)"
    echo "=== $(date '+%H:%M:%S') finished $name ==="
}

# --- A. Kvasir weight-decay probe (baseline wd=1e-5 already has 3-5 seeds) ---
run run_seg.py kv_attn_ln_wd1e-4  $KV --head pyramidal_attention --level-norm --weight-decay 1e-4 --seeds 0 1 2
run run_seg.py kv_single19_wd1e-4 $KV --head pyramidal_single_level --channels 19 --weight-decay 1e-4 --seeds 0 1 2
run run_seg.py kv_attn_ln_wd1e-3  $KV --head pyramidal_attention --level-norm --weight-decay 1e-3 --seeds 0 1 2
run run_seg.py kv_single19_wd1e-3 $KV --head pyramidal_single_level --channels 19 --weight-decay 1e-3 --seeds 0 1 2

# --- C1. CIFAR-100 to 5 seeds (largest, most significant parity win) ---
run run_cls.py c100_plain_s34     $C100 --head plain_conv --seeds 3 4
run run_cls.py c100_single_s34    $C100 --head pyramidal_single_level --seeds 3 4
run run_cls.py c100_pyr_s34       $C100 --head pyramidal --seeds 3 4
run run_cls.py c100_attn_s34      $C100 --head pyramidal_attention --seeds 3 4

# --- B. Kvasir third parity bracket: pyramidal+ln ch17 (21593) vs single ch18 (21259) ---
run run_seg.py kv_pyr_ln_ch17     $KV --head pyramidal --level-norm --channels 17 --seeds 0 1 2
run run_seg.py kv_single18        $KV --head pyramidal_single_level --channels 18 --seeds 0 1 2

# --- C2. CIFAR-10 parity pair to 10 seeds (5 seeds gave +0.7 pts at p=0.14) ---
run run_cls.py c10_pyr_s5789      $C10 --head pyramidal --seeds 5 6 7 8 9
run run_cls.py c10_single18_s5789 $C10 --head pyramidal_single_level --channels 18 --seeds 5 6 7 8 9

echo "=== iteration-6 batch complete ==="
