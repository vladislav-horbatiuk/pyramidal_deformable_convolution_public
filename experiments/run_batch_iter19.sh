#!/usr/bin/env bash
# Iteration-19 batch (~4 h on CPU, two parallel streams).
#
# CIFAR-100 300-epoch pair, pyramidal 95 991 vs width-matched single ch17
# 97 880, extended from seeds 0-2 to 0-9 per arm. At n=3 the gap (+0.008,
# p=0.25) was unresolved; n=10 gives ~80 % power for a 0.008 effect at a
# per-seed sd of ~0.006-0.009. The seed count is fixed in advance: report the
# n=10/10 result whatever it shows.
mkdir -p results/logs

C100="--dataset cifar100 --epochs 300 --num-workers 6"

run() {
    local script="$1"; local name="$2"; shift 2
    echo "=== $(date '+%H:%M:%S') starting $name ==="
    micromamba run -n ml_env python3 "experiments/${script}" "$@" \
        > "results/logs/iter19_${name}.log" 2>&1 \
        || echo "!!! $name FAILED (see results/logs/iter19_${name}.log)"
    echo "=== $(date '+%H:%M:%S') finished $name ==="
}

# Two streams in parallel. run_cls.py appends to classification.csv once,
# after its last seed, so the two writes are ~minutes apart; per-seed results
# are also in each log's JSON lines if a stream dies early.
run run_cls.py c100_pyr_ep300_s3-9 $C100 --head pyramidal --seeds 3 4 5 6 7 8 9 &
run run_cls.py c100_single_ch17_ep300_s3-9 $C100 --head pyramidal_single_level --channels 17 --seeds 3 4 5 6 7 8 9 &
wait

echo "=== iteration-19 batch complete ==="
