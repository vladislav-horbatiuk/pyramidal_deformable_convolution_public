#!/usr/bin/env bash
# Iteration-17 batch (~3-3.5 h on CPU, sequential).
#
# CIFAR-100 capacity-parity fix. The CIFAR-100 spatial-only control was only
# ever run at ch16 (93 870 params), below the full-z PDC head (95 991) and the
# attention head (99 906), so "beats a larger baseline" was not true there.
# Widen the control until it holds at least as many parameters:
#   ch17 = 97 880 >= full-z 95 991
#   ch18 = 101 962 >= attention 99 906
# Same protocol as the existing CIFAR-100 seeds 0-4 (150 epochs, default lr/wd).
mkdir -p results/logs

C100="--dataset cifar100 --epochs 150 --num-workers 6"

run() {
    local script="$1"; local name="$2"; shift 2
    echo "=== $(date '+%H:%M:%S') starting $name ==="
    micromamba run -n ml_env python3 "experiments/${script}" "$@" \
        > "results/logs/iter17_${name}.log" 2>&1 \
        || echo "!!! $name FAILED (see results/logs/iter17_${name}.log)"
    echo "=== $(date '+%H:%M:%S') finished $name ==="
}

run run_cls.py c100_single_ch17 $C100 --head pyramidal_single_level --channels 17 --seeds 0 1 2 3 4
run run_cls.py c100_single_ch18 $C100 --head pyramidal_single_level --channels 18 --seeds 0 1 2 3 4

echo "=== iteration-17 batch complete ==="
