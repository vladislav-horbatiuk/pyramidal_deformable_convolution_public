#!/usr/bin/env bash
# Iteration-21 batch: depth stretch-and-clip, seeds 1-4 (~4.5 h, CPU).
#
# Completes the 5-seed paired test started in iteration 20 (seed 0). Arm:
# cheap-z + LN + --z-stretch 0.1, paired by seed with the existing cheap-z + LN
# seeds 0-4 on both datasets. Judged on seed-paired Dice and training-loss
# tail; level mass on the end levels is the mechanism check. Seed 0 showed
# weight moving towards the end levels (Oxford reference 0.3% -> 13%, Kvasir
# shallowest 5.8% -> 9.8%) with small positive Dice/fit deltas inside noise.
mkdir -p results/logs

OX="--dataset oxford_pet --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
KV="--dataset kvasir --size 96 96 --batch-size 64 --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
CHEAP="--head pyramidal --level-norm --z-shared"

run() {
    local script="$1"; local name="$2"; shift 2
    echo "=== $(date '+%H:%M:%S') starting $name ==="
    micromamba run -n ml_env python3 "experiments/${script}" "$@" \
        > "results/logs/iter21_${name}.log" 2>&1 \
        || echo "!!! $name FAILED (see results/logs/iter21_${name}.log)"
    echo "=== $(date '+%H:%M:%S') finished $name ==="
}

run run_seg.py kv_zstretch_s1-4 $KV $CHEAP --z-stretch 0.1 --seeds 1 2 3 4
run run_seg.py ox_zstretch_s1-4 $OX $CHEAP --z-stretch 0.1 --seeds 1 2 3 4

echo "=== iteration-21 batch complete ==="
