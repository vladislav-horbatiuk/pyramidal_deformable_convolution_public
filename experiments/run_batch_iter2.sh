#!/usr/bin/env bash
# Iteration-2 batch (Oxford-IIIT Pet, CPU, sequential; ~13.5h):
#   A. capacity-matched controls (baselines get >= params of pyramidal heads)
#   B. fixed-z ablation (learned vs frozen scale selection)
#   C. attention z_init=last ablation
#   D. seeds 2-4 for the core heads (5 seeds total for significance tests)
mkdir -p results/logs

COMMON="--dataset oxford_pet --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"

run() {
    local name="$1"; shift
    echo "=== $(date '+%H:%M:%S') starting $name ==="
    micromamba run -n ml_env python3 experiments/run_seg.py $COMMON "$@" \
        > "results/logs/iter2_${name}.log" 2>&1 \
        || echo "!!! $name FAILED (see results/logs/iter2_${name}.log)"
    echo "=== $(date '+%H:%M:%S') finished $name ==="
}

# A. capacity-matched controls
run plain_ch19        --head plain_conv --channels 19 --seeds 0 1
run plain_ch21        --head plain_conv --channels 21 --seeds 0 1
run single_ch17       --head pyramidal_single_level --channels 17 --seeds 0 1
run single_ch19       --head pyramidal_single_level --channels 19 --seeds 0 1

# B. fixed-z ablation
run pyramidal_freezez --head pyramidal --freeze-z --seeds 0 1

# C. attention z_init ablation
run attention_zlast   --head pyramidal_attention --z-init last --seeds 0 1

# D. extra seeds for significance (core heads, default width)
run plain_s234        --head plain_conv --seeds 2 3 4
run single_s234       --head pyramidal_single_level --seeds 2 3 4
run pyramidal_s234    --head pyramidal --seeds 2 3 4
run attention_s234    --head pyramidal_attention --seeds 2 3 4

echo "=== iteration-2 batch complete ==="
