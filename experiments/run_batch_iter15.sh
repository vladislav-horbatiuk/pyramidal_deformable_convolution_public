#!/usr/bin/env bash
# Iteration-15 mini-batch (~2.5 h): Kvasir ~23k bracket to n=8 per arm.
#
# At n=6/6 the attention+LN val deficit sits at seeds p=0.060 (RESULTS S8)
# while its *fitting* deficit is already significant (p=0.031, S6). Two more
# seeds per arm settle the val side either way; at the observed D=-0.018 and
# sd~0.011/0.008 the power analysis puts 80% power at ~8/arm.
mkdir -p results/logs

KV="--dataset kvasir --size 96 96 --batch-size 64 --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"

run() {
    local name="$1"; shift
    echo "=== $(date '+%H:%M:%S') starting $name ==="
    micromamba run -n ml_env python3 experiments/run_seg.py "$@" \
        > "results/logs/iter15_${name}.log" 2>&1 \
        || echo "!!! $name FAILED (see results/logs/iter15_${name}.log)"
    echo "=== $(date '+%H:%M:%S') finished $name ==="
}

run kv_attn_ln_s67      $KV --head pyramidal_attention --level-norm --seeds 6 7
run kv_single_ch19_s67  $KV --head pyramidal_single_level --channels 19 --seeds 6 7

echo "=== iteration-15 batch complete ==="
