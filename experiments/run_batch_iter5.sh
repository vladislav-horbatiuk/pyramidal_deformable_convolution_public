#!/usr/bin/env bash
# Iteration-5 consolidation batch (~14h CPU, sequential): bring every headline
# table row to 5 seeds (Oxford, CIFAR) / 5 seeds (Kvasir core), and add the
# missing Kvasir width control (single ch19 vs attention+lvlnorm 23.3k).
mkdir -p results/logs

OX="--dataset oxford_pet --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
KV="--dataset kvasir --size 96 96 --batch-size 64 --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
C10="--dataset cifar10 --epochs 150 --num-workers 6"

run() {
    local script="$1" name="$2"; shift 2
    echo "=== $(date '+%H:%M:%S') starting $name ==="
    micromamba run -n ml_env python3 "experiments/${script}" "$@" \
        > "results/logs/iter5_${name}.log" 2>&1 \
        || echo "!!! $name FAILED (see results/logs/iter5_${name}.log)"
    echo "=== $(date '+%H:%M:%S') finished $name ==="
}

# Kvasir: missing width control + 5th/4th seeds for core rows (cheap first)
run run_seg.py kv_single_ch19    $KV --head pyramidal_single_level --channels 19 --seeds 0 1 2
run run_seg.py kv_plain_s34      $KV --head plain_conv --seeds 3 4
run run_seg.py kv_single_s34     $KV --head pyramidal_single_level --seeds 3 4
run run_seg.py kv_single17_s34   $KV --head pyramidal_single_level --channels 17 --seeds 3 4
run run_seg.py kv_pyr_ln_s34     $KV --head pyramidal --level-norm --seeds 3 4
run run_seg.py kv_attn_ln_s34    $KV --head pyramidal_attention --level-norm --seeds 3 4

# CIFAR-10: seeds 3-4 for headline rows
run run_cls.py c10_plain_s34     $C10 --head plain_conv --seeds 3 4
run run_cls.py c10_single18_s34  $C10 --head pyramidal_single_level --channels 18 --seeds 3 4
run run_cls.py c10_pyr_s34       $C10 --head pyramidal --seeds 3 4
run run_cls.py c10_attn_s34      $C10 --head pyramidal_attention --seeds 3 4

# Oxford: seeds 2-4 for the capacity controls and lvlnorm heads
run run_seg.py ox_single17_s234  $OX --head pyramidal_single_level --channels 17 --seeds 2 3 4
run run_seg.py ox_single19_s234  $OX --head pyramidal_single_level --channels 19 --seeds 2 3 4
run run_seg.py ox_pyr_ln_s234    $OX --head pyramidal --level-norm --seeds 2 3 4
run run_seg.py ox_attn_ln_s234   $OX --head pyramidal_attention --level-norm --seeds 2 3 4

echo "=== iteration-5 batch complete ==="
