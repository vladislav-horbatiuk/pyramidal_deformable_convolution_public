#!/usr/bin/env bash
# Iteration-10 batch (~5 h, sequential, priority-ordered).
#
# One job: resolve the cheap-z lead, which is now the study's only live
# positive on Kvasir and a design rule the paper currently states incorrectly.
#
# Standing at 5 seeds (iteration 9), pyramidal+level-norm+z-shared (18,260):
#   Kvasir  vs single ch17     (19,161)  +0.0131  p=0.020
#           vs single+LN ch17  (19,195)  +0.0071  p=0.070   <- the marginal row
#           vs trilinear+LN    (19,420)  +0.0092  p=0.004
#   Oxford  vs single+LN ch17  (19,195)  +0.0024  p=6e-16  (only 3 seeds)
#
#   A (~1.5 h) Kvasir cheap-z seeds 5-9 -> n=10. The p=0.070 row is the whole
#              question: is cheap-z genuinely at parity-or-better on the dataset
#              where every other configuration ties or loses? Kvasir's per-seed
#              spread is large (sd .0072-.0181 across configs in this bracket),
#              so n=5 is thin for a +0.007 effect.
#   B (~1 h)   Its control to n=10 as well. Iteration 9 already showed this
#              control moving .6182 -> .6307 between n=3 and n=5, which is what
#              turned p=0.004 into p=0.070. Growing only the arm would repeat
#              exactly that mistake in the other direction.
#   C (~2 h)   Oxford cheap-z seeds 3-4 -> n=5, matching every other headline
#              row on that dataset.
#
# Deliberately NOT extending attention+LN+z-shared: at +0.0003 (p=0.80) against
# a larger control, with the probe showing its level weights are 96% a global
# constant, more seeds would only measure a null more precisely.
mkdir -p results/logs

OX="--dataset oxford_pet --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
KV="--dataset kvasir --size 96 96 --batch-size 64 --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"

run() {
    local script="$1" name="$2"; shift 2
    echo "=== $(date '+%H:%M:%S') starting $name ==="
    micromamba run -n ml_env python3 "experiments/${script}" "$@" \
        > "results/logs/iter10_${name}.log" 2>&1 \
        || echo "!!! $name FAILED (see results/logs/iter10_${name}.log)"
    echo "=== $(date '+%H:%M:%S') finished $name ==="
}

# --- A. the arm ------------------------------------------------------------
run run_seg.py kv_pyr_ln_zshared_s5789 $KV --head pyramidal --level-norm --z-shared --seeds 5 6 7 8 9

# --- B. the control, same seeds --------------------------------------------
run run_seg.py kv_single_ln_ch17_s5789 $KV --head pyramidal_single_level --level-norm --channels 17 --seeds 5 6 7 8 9

# --- C. Oxford to 5 seeds --------------------------------------------------
run run_seg.py ox_pyr_ln_zshared_s34   $OX --head pyramidal --level-norm --z-shared --seeds 3 4

echo "=== iteration-10 batch complete ==="
