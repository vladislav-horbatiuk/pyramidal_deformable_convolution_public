#!/usr/bin/env bash
# Iteration-11 batch (~1.5 h). One loose end.
#
# Iteration 10 established the study's first significant capacity-parity win on
# Kvasir: pyramidal+level-norm+z-shared (18,260) vs single_level+LN ch17
# (19,195), +0.0095, p=0.009, both at n=10, control holding 935 more params.
#
# The other ~19.2k control -- single_level ch17 WITHOUT level-norm (19,161) --
# gives the same effect size (+0.0099) but only p=0.099, purely because it sits
# at n=5 against the arm's n=10. The two controls are within 0.0004 of each
# other in macro Dice (.6251 vs .6247), so this is a power gap, not a
# disagreement. Closing it lets the win be quoted against whichever control
# turns out stronger, at full power, instead of inviting the question.
mkdir -p results/logs

KV="--dataset kvasir --size 96 96 --batch-size 64 --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"

run() {
    local script="$1" name="$2"; shift 2
    echo "=== $(date '+%H:%M:%S') starting $name ==="
    micromamba run -n ml_env python3 "experiments/${script}" "$@" \
        > "results/logs/iter11_${name}.log" 2>&1 \
        || echo "!!! $name FAILED (see results/logs/iter11_${name}.log)"
    echo "=== $(date '+%H:%M:%S') finished $name ==="
}

run run_seg.py kv_single_ch17_s5789 $KV --head pyramidal_single_level --channels 17 --seeds 5 6 7 8 9

echo "=== iteration-11 batch complete ==="
