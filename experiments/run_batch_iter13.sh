#!/usr/bin/env bash
# Iteration-13 batch (~11 h, sequential, priority-ordered).
#
# Iteration 12's structured-offset results were all inside seed noise: every
# comparison that the image-paired Wilcoxon called significant (p down to 3e-8)
# collapsed to p=0.36-0.87 under a seed-level Welch test. This batch brings the
# structured variants to the same seed count as the cheap-z baseline so the
# comparison can be made at the level that actually matters.
#
#   A (~5.6 h) Kvasir similarity ch17 and affine ch17 to n=10, matching cheap-z.
#   B (~5.3 h) Oxford all three structured variants to n=5, matching cheap-z.
#
# POWER WARNING -- read before interpreting the result. Kvasir per-seed sd in
# this bracket is ~0.010 micro Dice, so at n=10 vs n=10 the standard error of a
# difference is 0.010*sqrt(2/10) = 0.0045. That resolves effects of ~0.010 at
# about 2 sigma and leaves anything at 0.005 unresolvable. Oxford is worse in
# relative terms: sd ~0.0025 at n=5 gives SE 0.0016, so the +0.002 differences
# iteration 12 reported would need n~25 per arm to reach p<0.05.
#
# In other words this batch can confirm or reject a ~0.01 effect and nothing
# smaller. If everything lands inside +-0.005 again, the correct conclusion is
# "structured offsets are indistinguishable from free offsets at the seed
# budget we can afford", and further seeds are not the answer -- a larger
# dataset or a bigger backbone would be.
mkdir -p results/logs

OX="--dataset oxford_pet --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
KV="--dataset kvasir --size 96 96 --batch-size 64 --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
CHEAP="--head pyramidal --level-norm --z-shared"

run() {
    local name="$1"; shift
    echo "=== $(date '+%H:%M:%S') starting $name ==="
    micromamba run -n ml_env python3 experiments/run_seg.py "$@" \
        > "results/logs/iter13_${name}.log" 2>&1 \
        || echo "!!! $name FAILED (see results/logs/iter13_${name}.log)"
    echo "=== $(date '+%H:%M:%S') finished $name ==="
}

# --- A. Kvasir to n=10 (cheap-z is already there) ---------------------------
run kv_similarity_ch17_s3_9 $KV $CHEAP --offset-mode similarity --channels 17 --seeds 3 4 5 6 7 8 9
run kv_affine_ch17_s3_9     $KV $CHEAP --offset-mode affine     --channels 17 --seeds 3 4 5 6 7 8 9

# --- B. Oxford to n=5 (cheap-z is already there) -----------------------------
run ox_affine_ch17_s34      $OX $CHEAP --offset-mode affine     --channels 17 --seeds 3 4
run ox_affine_ch16_s34      $OX $CHEAP --offset-mode affine                   --seeds 3 4
run ox_similarity_ch17_s34  $OX $CHEAP --offset-mode similarity --channels 17 --seeds 3 4

echo "=== iteration-13 batch complete ==="
