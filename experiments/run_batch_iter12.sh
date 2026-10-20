#!/usr/bin/env bash
# Iteration-12 batch (~10 h, sequential, priority-ordered).
#
# Tests the rungs between the reduced block (a pure translation of the sampling
# window) and cheap-z (18 free spatial numbers per location).
#
# Motivation, from decomposing trained cheap-z / full-z checkpoints as
# Delta_k = t + A p_k + residual:
#
#              translation   affine   free-form residual
#   Kvasir       15-24%      46-47%       30-38%
#   Oxford       25-27%      27-28%       44-48%
#
# Translation alone -- all the reduced block can express -- captures 15-27% of
# what the operator actually does, which is why that variant is dominated.
# Adding a 2x2 linear map (6 numbers instead of 18) captures 54-70%.
#
# The dilation component is also strongly coupled to the depth branch on Kvasir
# (spearman(kernel dilation, z) = +0.75 / +0.76 across two configs, n=147k):
# where the block sits on a fine level it spreads the kernel, where it drops to
# a coarse level it stays compact -- two routes to the same physical receptive
# field. So the affine rung parameterises exactly the mode that interacts with
# the mechanism this paper is about.
#
# Prediction under test: fewer degrees of freedom should help where data is
# scarce and hurt where it is plentiful -- the same trade that made cheap-z beat
# full-z. Kvasir is affine-dominated (46%), Oxford is residual-dominated (48%),
# so affine should be competitive on Kvasir and lose on Oxford.
#
#   A (~3.5 h) affine at cheap-z's own parameter count (ch17, 18 521 vs 18 260 --
#              260 MORE, so a win is not bought with parameters). The direct
#              degrees-of-freedom comparison.
#   B (~3.5 h) affine at ch16 (16 520): 1 740 FEWER parameters than cheap-z.
#              Asks the parameter-efficiency question rather than the dof one.
#   C (~3 h)   similarity at ch17 (18 213, again matched): scale+rotation only,
#              4 numbers. If affine works, does the shear/anisotropy part of it
#              matter, or is isotropic dilation the whole story?
#
# Baselines already on disk at this bracket (macro Dice):
#   Kvasir  cheap-z 18 260 .6346 (n10) | single ch17 19 161 .6252 (n10)
#           single+LN ch17 19 195 .6251 (n10) | full-z 19 420 .6285 (n5)
#   Oxford  cheap-z 18 260 .8533 (n5)  | single+LN ch17 19 195 .8512 (n5)
mkdir -p results/logs

OX="--dataset oxford_pet --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
KV="--dataset kvasir --size 96 96 --batch-size 64 --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
CHEAP="--head pyramidal --level-norm --z-shared"

run() {
    local name="$1"; shift
    echo "=== $(date '+%H:%M:%S') starting $name ==="
    micromamba run -n ml_env python3 experiments/run_seg.py "$@" \
        > "results/logs/iter12_${name}.log" 2>&1 \
        || echo "!!! $name FAILED (see results/logs/iter12_${name}.log)"
    echo "=== $(date '+%H:%M:%S') finished $name ==="
}

# --- A. affine at matched capacity -----------------------------------------
run kv_affine_ch17  $KV $CHEAP --offset-mode affine --channels 17 --seeds 0 1 2
run ox_affine_ch17  $OX $CHEAP --offset-mode affine --channels 17 --seeds 0 1 2

# --- B. affine at its natural (cheaper) width -------------------------------
run kv_affine_ch16  $KV $CHEAP --offset-mode affine --seeds 0 1 2
run ox_affine_ch16  $OX $CHEAP --offset-mode affine --seeds 0 1 2

# --- C. similarity: is isotropic dilation the whole story? -------------------
run kv_similarity_ch17 $KV $CHEAP --offset-mode similarity --channels 17 --seeds 0 1 2
run ox_similarity_ch17 $OX $CHEAP --offset-mode similarity --channels 17 --seeds 0 1 2

echo "=== iteration-12 batch complete ==="
