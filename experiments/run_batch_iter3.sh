#!/usr/bin/env bash
# Iteration-3 batch (~14h CPU, sequential):
#   A. Kvasir-SEG (real scale variation) — full head set incl. reduced family,
#      with capacity control (single ch17 ~ pyramidal params), 3 seeds
#   B. Oxford Pet reduced-family: seeds 2-4, reduced_single_level control
#      (14.9k params ~ plain_conv 14.6k), no-adapters ablation, plain ch17
#   C. Oxford Pet cheap-z variants (z shared across kernel points; 18.1k/18.6k
#      params — below the single ch17 19.2k control)
#   D. Oxford Pet skip-less U-Net (head pyramid = only multi-scale path);
#      single ch20 noskip (18.6k) capacity-matches attention noskip (18.6k)
mkdir -p results/logs

KVASIR="--dataset kvasir --size 96 96 --batch-size 64 --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
OXFORD="--dataset oxford_pet --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"

run() {
    local name="$1"; shift
    echo "=== $(date '+%H:%M:%S') starting $name ==="
    micromamba run -n ml_env python3 experiments/run_seg.py "$@" \
        > "results/logs/iter3_${name}.log" 2>&1 \
        || echo "!!! $name FAILED (see results/logs/iter3_${name}.log)"
    echo "=== $(date '+%H:%M:%S') finished $name ==="
}

# A. Kvasir-SEG, cheapest first
run kvasir_plain        $KVASIR --head plain_conv --seeds 0 1 2
run kvasir_reduced_single $KVASIR --head pyramidal_reduced_single_level --seeds 0 1 2
run kvasir_reduced      $KVASIR --head pyramidal_reduced --seeds 0 1 2
run kvasir_single       $KVASIR --head pyramidal_single_level --seeds 0 1 2
run kvasir_single_ch17  $KVASIR --head pyramidal_single_level --channels 17 --seeds 0 1 2
run kvasir_pyramidal    $KVASIR --head pyramidal --seeds 0 1 2
run kvasir_attention    $KVASIR --head pyramidal_attention --seeds 0 1 2

# B. Oxford Pet reduced family
run oxford_reduced_single $OXFORD --head pyramidal_reduced_single_level --seeds 0 1 2 3 4
run oxford_reduced_s234   $OXFORD --head pyramidal_reduced --seeds 2 3 4
run oxford_reduced_noadapt $OXFORD --head pyramidal_reduced --no-adapters --seeds 0 1
run oxford_plain_ch17     $OXFORD --head plain_conv --channels 17 --seeds 0 1

# C. Oxford Pet cheap-z
run oxford_pyramidal_zshared $OXFORD --head pyramidal --z-shared --seeds 0 1
run oxford_attention_zshared $OXFORD --head pyramidal_attention --z-shared --seeds 0 1

# D. Oxford Pet skip-less
run oxford_plain_noskip      $OXFORD --head plain_conv --no-skips --seeds 0 1
run oxford_single20_noskip   $OXFORD --head pyramidal_single_level --channels 20 --no-skips --seeds 0 1
run oxford_attention_noskip  $OXFORD --head pyramidal_attention --no-skips --seeds 0 1

echo "=== iteration-3 batch complete ==="
