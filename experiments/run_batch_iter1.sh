#!/usr/bin/env bash
# Iteration-1 head comparison on Oxford-IIIT Pet (CPU).
# Runs sequentially (cheapest first) so per-run timings stay comparable.
mkdir -p results/logs

HEADS="plain_conv multiscale_fusion pyramidal_reduced pyramidal_single_level pyramidal pyramidal_attention"

for head in $HEADS; do
    echo "=== $(date '+%H:%M:%S') starting $head ==="
    micromamba run -n ml_env python3 experiments/run_seg.py \
        --dataset oxford_pet --head "$head" \
        --seeds 0 1 --epochs 60 --eval-every 5 \
        --num-workers 6 --save-models \
        > "results/logs/iter1_${head}.log" 2>&1 \
        || echo "!!! $head FAILED (see results/logs/iter1_${head}.log)"
    echo "=== $(date '+%H:%M:%S') finished $head ==="
done
echo "=== batch complete ==="
