#!/usr/bin/env bash
# Iteration-7 batch: four targeted interventions, each aimed at a specific
# finding from the artifact analysis (results/RESULTS.md).
#
# Ordered by expected value, sequentially, so stopping early still leaves the
# most informative results on disk. Cumulative wall-clock estimates in the
# section headers assume an otherwise-idle machine -- if the Cityscapes
# plain-ch68 run is going at the same time, expect roughly double.
#
#   A (~2.5 h) Reduced block + level-norm. The reduced variant was last run in
#              iteration 3 and never got the intervention that overturned the
#              full block's verdict -- it structurally could not, having had no
#              use_level_norm until now. Its negative Kvasir result is
#              confounded exactly as the full block's iteration-2/3 results
#              were. Also the cheapest existing instance of "make pyramid
#              access nearly free", which is what Kvasir's width ladder demands.
#   B (~3 h)   Residual cross-scale on Kvasir. §10.1 shows trilinear+LN puts
#              ~0.01 of its sampling weight on the reference level: the block
#              *trades away* full resolution to buy context. The residual form
#              starts as an exact DCN (gate initialised to 0) and can only add,
#              so it should not be able to lose to the DCN control.
#   C (~3 h)   Depth-branch weight init on CIFAR-10, where the pyramid's seed
#              variance is 3.4x the control's and the parity claim is only
#              directional. Zero-init makes the z branch a constant function at
#              step 0, which has to break symmetry through gradients alone.
#   D (~9 h)   Residual cross-scale on Oxford: does it cost anything where the
#              pyramid already wins? Plus the DCNv2 modulation mask, which is
#              implemented but has never been tested at parity.
#
# Measured overhead of --residual-cross-scale after routing its reference
# branch through sample_one_level instead of the full L-level trilinear pass:
# +14% per step (33.9s vs 29.7s for one Kvasir epoch), not the +93% it cost
# before that fix.
#
# Capacity notes: the residual gate is one parameter per input channel (+16),
# so rescross rows are matched against the same controls as their non-residual
# counterparts. reduced+LN adds 2*C per level (+128). attn+LN+mask at 24,640
# needs the ch20 single-level control (25,779), which is why D runs it.
mkdir -p results/logs

OX="--dataset oxford_pet --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
KV="--dataset kvasir --size 96 96 --batch-size 64 --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
C10="--dataset cifar10 --epochs 150 --num-workers 6"

run() {
    local script="$1" name="$2"; shift 2
    echo "=== $(date '+%H:%M:%S') starting $name ==="
    micromamba run -n ml_env python3 "experiments/${script}" "$@" \
        > "results/logs/iter7_${name}.log" 2>&1 \
        || echo "!!! $name FAILED (see results/logs/iter7_${name}.log)"
    echo "=== $(date '+%H:%M:%S') finished $name ==="
}

# --- A. reduced block, with and without level-norm --------------------------
# Controls already on disk: reduced ch16 (15,812, 3 seeds), reduced_single ch16
# (14,851, 3+ seeds), plain ch16 (14,561, 5 seeds).
run run_seg.py kv_reduced_ln       $KV --head pyramidal_reduced --level-norm --seeds 0 1 2
run run_seg.py ox_reduced_ln       $OX --head pyramidal_reduced --level-norm --seeds 0 1 2
# same intervention on the reduced single-level control, so the pyramid-access
# comparison stays like-for-like rather than comparing normed vs un-normed
run run_seg.py kv_reduced_single_ln $KV --head pyramidal_reduced_single_level --level-norm --seeds 0 1 2
run run_seg.py ox_reduced_single_ln $OX --head pyramidal_reduced_single_level --level-norm --seeds 0 1 2

# --- B. residual cross-scale, Kvasir (the dataset the block loses) ----------
run run_seg.py kv_tri_ln_rescross  $KV --head pyramidal --level-norm --residual-cross-scale --seeds 0 1 2
run run_seg.py kv_attn_ln_rescross $KV --head pyramidal_attention --level-norm --residual-cross-scale --seeds 0 1 2

# --- C. depth-branch init, CIFAR-10 -----------------------------------------
run run_cls.py c10_pyr_zwstd       $C10 --head pyramidal --z-weight-init-std 0.01 --seeds 0 1 2 3 4
run run_cls.py c10_attn_zwstd      $C10 --head pyramidal_attention --z-weight-init-std 0.01 --seeds 0 1 2 3 4

# --- D. residual on Oxford + the untested modulation mask -------------------
run run_seg.py ox_tri_ln_rescross  $OX --head pyramidal --level-norm --residual-cross-scale --seeds 0 1 2
run run_seg.py ox_attn_ln_rescross $OX --head pyramidal_attention --level-norm --residual-cross-scale --seeds 0 1 2
run run_seg.py kv_attn_ln_mask     $KV --head pyramidal_attention --level-norm --use-mask --seeds 0 1 2
run run_seg.py kv_single_ch20      $KV --head pyramidal_single_level --channels 20 --seeds 0 1 2

echo "=== iteration-7 batch complete ==="
