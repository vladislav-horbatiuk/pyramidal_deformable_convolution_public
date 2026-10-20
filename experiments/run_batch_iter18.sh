#!/usr/bin/env bash
# Iteration-18 batch (~1.7 h on CPU, sequential).
#
# Follow-up to iteration 17 (CIFAR-100 width-matched controls). Two items still
# compared the pyramid against the under-sized ch16 control:
# A  ch17 seed-0 checkpoint for the per-class / difficulty-bin analysis
#    (iteration 17 did not save models). Same flags as iteration 17, so the
#    rerun must reproduce its seed-0 accuracy (.4348) exactly.
# B  The 300-epoch pair: pyramidal 95 991 vs single ch17 97 880, seeds 0-2,
#    matching the existing pyramidal ep=300 seeds.
mkdir -p results/logs

C100="--dataset cifar100 --num-workers 6"

run() {
    local script="$1"; local name="$2"; shift 2
    echo "=== $(date '+%H:%M:%S') starting $name ==="
    micromamba run -n ml_env python3 "experiments/${script}" "$@" \
        > "results/logs/iter18_${name}.log" 2>&1 \
        || echo "!!! $name FAILED (see results/logs/iter18_${name}.log)"
    echo "=== $(date '+%H:%M:%S') finished $name ==="
}

# --- A. ch17 checkpoint ------------------------------------------------------
run run_cls.py c100_single_ch17_ckpt $C100 --epochs 150 --head pyramidal_single_level --channels 17 --save-models --seeds 0

# --- B. 300-epoch matched control --------------------------------------------
run run_cls.py c100_single_ch17_ep300 $C100 --epochs 300 --head pyramidal_single_level --channels 17 --seeds 0 1 2

echo "=== iteration-18 batch complete ==="
