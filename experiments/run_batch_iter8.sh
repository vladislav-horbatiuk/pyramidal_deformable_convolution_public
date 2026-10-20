#!/usr/bin/env bash
# Iteration-8 batch (~10.5 h, sequential, priority-ordered so an early stop
# still leaves the most consequential results on disk).
#
# Iteration 7 produced one win (reduced block + level-norm beats its
# single-level control on Kvasir, +0.028, p=3e-8) and exposed one confound.
# This batch settles both, and asks why the reduced block succeeds where the
# full block fails.
#
#   A (~4 h)  THE NORMALIZED DCN CONTROL. Every capacity-parity claim in
#             results/RESULTS.md §4 compares pyramid+level-norm against a
#             single_level control WITHOUT level-norm. Iteration 7 showed that
#             adding level-norm to reduced_single_level -- which has no pyramid
#             access at all (num_levels=1, so its "level-norm" is just a
#             GroupNorm before the deformable conv) -- gained it ~+0.011 on
#             Oxford, enough to erase that family's previously significant
#             +0.0047 win. If a plain normalizer is worth that much on its own,
#             part of what we attribute to cross-scale access is normalisation.
#             At risk: Oxford +0.0039 (p=5e-26) and +0.0015 (p=3e-4), both
#             smaller than the gain reduced_single got. Oxford first because
#             that is where the wins are; Kvasir in D since its verdicts are
#             already ties/losses and a stronger control only confirms them.
#   B (~1 h)  CAPACITY HONESTY FOR THE NEW WIN. reduced+LN (15,940) was compared
#             against reduced_single+LN (14,883) -- the pyramid arm holds 7.1%
#             MORE parameters, which violates the protocol used everywhere else.
#             reduced_single+LN ch17 (16,731) is the honest control. Kvasir's
#             ladder is ~0.44 Dice pts/1k params so 1,057 params is worth ~0.005
#             against an observed +0.028, but that should be measured.
#   C (~1 h)  WHY DOES REDUCED WIN WHERE FULL LOSES? The reduced block predicts
#             3 offset channels, the full block 27. Probe evidence says the
#             offset branch is over-parameterised (per-kernel-point z is not
#             load-bearing, the z field is spatially smooth, attention is 27% a
#             global constant). --z-shared cuts only the z part (27->19
#             channels, 18,260) and has NEVER been run on Kvasir -- z-sharing
#             was established on Oxford, pre-level-norm. If it also wins, the
#             effect is graded with offset-branch cost; if only the fully
#             reduced block wins, it is the architecture.
#   D (~3 h)  Kvasir normalized control; reduced block on CIFAR (never run
#             there at all, and CIFAR-100 is the strongest positive in the
#             study); seeds 3-4 to bring reduced+LN to 5.
#
# Capacity brackets (control must hold >= params of the arm it tests):
#   single_level+LN ch17 19,195  vs  trilinear+LN   19,420   <- A, arm is larger
#   single_level+LN ch19 23,503  vs  attention+LN   23,335
#   reduced_single+LN ch17 16,731 vs reduced+LN     15,940
#   single_level ch17    19,161  vs  pyr+LN+zshared 18,260
#   cls reduced_single ch17 82,006 vs cls reduced   79,101
# The single_level+LN ch17 row is the one exception: 19,195 < 19,420, so a
# trilinear win there would still be worth 225 params of doubt. ch18 would
# overshoot to 21,3xx; ch17 is the closest honest bracket and the residual gap
# (~0.1 Dice pts by the Kvasir ladder) is far below the effect under test.
mkdir -p results/logs

OX="--dataset oxford_pet --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
KV="--dataset kvasir --size 96 96 --batch-size 64 --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
C10="--dataset cifar10 --epochs 150 --num-workers 6"

run() {
    local script="$1" name="$2"; shift 2
    echo "=== $(date '+%H:%M:%S') starting $name ==="
    micromamba run -n ml_env python3 "experiments/${script}" "$@" \
        > "results/logs/iter8_${name}.log" 2>&1 \
        || echo "!!! $name FAILED (see results/logs/iter8_${name}.log)"
    echo "=== $(date '+%H:%M:%S') finished $name ==="
}

# --- A. normalized DCN control, Oxford (the claims most at risk) ------------
run run_seg.py ox_single_ln_ch17   $OX --head pyramidal_single_level --level-norm --channels 17 --seeds 0 1 2
run run_seg.py ox_single_ln_ch19   $OX --head pyramidal_single_level --level-norm --channels 19 --seeds 0 1 2

# --- B. width-matched control for the iteration-7 reduced win ---------------
run run_seg.py kv_reduced_single_ln_ch17 $KV --head pyramidal_reduced_single_level --level-norm --channels 17 --seeds 0 1 2
run run_seg.py ox_reduced_single_ln_ch17 $OX --head pyramidal_reduced_single_level --level-norm --channels 17 --seeds 0 1 2

# --- C. is it the offset-branch cost? z-shared + level-norm on Kvasir -------
run run_seg.py kv_pyr_ln_zshared   $KV --head pyramidal --level-norm --z-shared --seeds 0 1 2

# --- D. Kvasir normalized control, reduced on CIFAR, extra seeds ------------
run run_seg.py kv_single_ln_ch17   $KV --head pyramidal_single_level --level-norm --channels 17 --seeds 0 1 2
run run_seg.py kv_single_ln_ch19   $KV --head pyramidal_single_level --level-norm --channels 19 --seeds 0 1 2
run run_cls.py c10_reduced         $C10 --head pyramidal_reduced --seeds 0 1 2 3 4
run run_cls.py c10_reduced_single17 $C10 --head pyramidal_reduced_single_level --channels 17 --seeds 0 1 2 3 4
run run_seg.py kv_reduced_ln_s34   $KV --head pyramidal_reduced --level-norm --seeds 3 4
run run_seg.py ox_reduced_ln_s34   $OX --head pyramidal_reduced --level-norm --seeds 3 4

echo "=== iteration-8 batch complete ==="
