#!/usr/bin/env bash
# Iteration-4 batch (~13h CPU, sequential):
#   A. CIFAR-10: capacity-bracketed head comparison, 3 seeds, 150 epochs
#      (params: plain 77.9k | single 80.5k | pyramidal 82.6k | single ch17
#       84.5k | attention 86.5k | single ch18 88.6k)
#   B. Kvasir + Oxford: per-level GroupNorm fusion trick (--level-norm) on
#      pyramidal/attention — the level-statistics hypothesis predicts the
#      biggest effect on Kvasir, where naive fusion hurt most
#   C. CIFAR-100: core heads, 3 seeds
mkdir -p results/logs

CLS10="--dataset cifar10 --epochs 150 --num-workers 6"
CLS100="--dataset cifar100 --epochs 150 --num-workers 6"
KV="--dataset kvasir --size 96 96 --batch-size 64 --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
OX="--dataset oxford_pet --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"

run() {
    local script="$1" name="$2"; shift 2
    echo "=== $(date '+%H:%M:%S') starting $name ==="
    micromamba run -n ml_env python3 "experiments/${script}" "$@" \
        > "results/logs/iter4_${name}.log" 2>&1 \
        || echo "!!! $name FAILED (see results/logs/iter4_${name}.log)"
    echo "=== $(date '+%H:%M:%S') finished $name ==="
}

# A. CIFAR-10
run run_cls.py c10_plain        $CLS10 --head plain_conv --seeds 0 1 2
run run_cls.py c10_single       $CLS10 --head pyramidal_single_level --seeds 0 1 2
run run_cls.py c10_single_ch17  $CLS10 --head pyramidal_single_level --channels 17 --seeds 0 1 2
run run_cls.py c10_single_ch18  $CLS10 --head pyramidal_single_level --channels 18 --seeds 0 1 2
run run_cls.py c10_pyramidal    $CLS10 --head pyramidal --seeds 0 1 2
run run_cls.py c10_attention    $CLS10 --head pyramidal_attention --seeds 0 1 2
run run_cls.py c10_pyr_lvlnorm  $CLS10 --head pyramidal --level-norm --seeds 0 1 2
run run_cls.py c10_attn_zshared $CLS10 --head pyramidal_attention --z-shared --seeds 0 1 2

# B. level-norm on segmentation (Kvasir first: strongest predicted effect)
run run_seg.py kv_pyr_lvlnorm   $KV --head pyramidal --level-norm --seeds 0 1 2
run run_seg.py kv_attn_lvlnorm  $KV --head pyramidal_attention --level-norm --seeds 0 1 2
run run_seg.py ox_pyr_lvlnorm   $OX --head pyramidal --level-norm --seeds 0 1
run run_seg.py ox_attn_lvlnorm  $OX --head pyramidal_attention --level-norm --seeds 0 1

# C. CIFAR-100 core heads
run run_cls.py c100_plain       $CLS100 --head plain_conv --seeds 0 1 2
run run_cls.py c100_single      $CLS100 --head pyramidal_single_level --seeds 0 1 2
run run_cls.py c100_pyramidal   $CLS100 --head pyramidal --seeds 0 1 2
run run_cls.py c100_attention   $CLS100 --head pyramidal_attention --seeds 0 1 2

echo "=== iteration-4 batch complete ==="
