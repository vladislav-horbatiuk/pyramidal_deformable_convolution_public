#!/usr/bin/env bash
# Local Cityscapes runs, one config per invocation:
#     bash experiments/run_cityscapes_local.sh <name> [extra flags...]
#
# Data was extracted from the official zips to ./cityscapes (train+val only,
# 2975/500; the test split is unlabelled and unused). The protocol below is
# locked to what Kaggle sessions 1-3 actually ran -- crop 256x256, batch 4,
# val-batch 1, base size 1024x512, 60 epochs, eval every 5 -- so local rows are
# directly comparable with the existing ones in results/cityscapes.csv.
#
# Measured local cost on this machine (24 cores, torch CPU), for planning:
#     plain_conv ch68             9.5 h
#     pyramidal_reduced+LN ch70  14.1 h
#     pyramidal_single_level ch66 42.5 h  (+LN: 43.8 h)
#     pyramidal ch64 +zs+LN      69.4 h
#     pyramidal_attention ch64   82.2 h
set -u
name="$1"; shift
mkdir -p results/logs results/cityscapes_local

COMMON="--root cityscapes --base-size 1024 512 --crop 256 256 \
--batch-size 4 --val-batch-size 1 --epochs 60 --eval-every 5 --num-workers 6 \
--out-dir results/cityscapes_local --results-csv results/cityscapes.csv"

echo "=== $(date '+%F %H:%M:%S') starting $name ==="
micromamba run -n ml_env python3 experiments/run_cityscapes.py $COMMON "$@" \
    > "results/logs/city_${name}.log" 2>&1 \
    || echo "!!! $name FAILED (see results/logs/city_${name}.log)"
echo "=== $(date '+%F %H:%M:%S') finished $name ==="
