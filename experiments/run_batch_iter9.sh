#!/usr/bin/env bash
# Iteration-9 batch (~8 h, sequential, priority-ordered).
#
# Iteration 8 left exactly one live lead and two thin control sets.
#
#   A (~4.5 h) CHEAP-Z, THE ONLY POSITIVE LEFT ON KVASIR. pyramidal+level-norm
#              +z-shared (18,260) came out ahead of every ~19.2k control, all of
#              which hold MORE parameters: +0.0087 (p=0.13) vs single ch17,
#              +0.0152 (p=0.004) vs single+LN ch17, +0.0049 (p=0.22) vs the
#              full-z block it is cheaper than. First configuration in the study
#              to be positive against a larger control on Kvasir -- but n=3, and
#              the significant row uses the normalised control, which on Kvasir
#              is the weaker baseline. Bring it to 5 seeds, test whether it
#              transfers to Oxford, and check whether attention benefits from
#              the same economy (18,695, also under every control).
#   B (~3.5 h) THICKEN THE CONTROLS. Iteration 8's corrected verdicts -- the
#              Oxford attention win dying (-0.0004, p=0.25) and the trilinear
#              win shrinking to +0.0016 -- both rest on n=3 normalised controls.
#              So does the reduced-family tie (-0.0055, p=0.13). Those are the
#              numbers the paper will now quote, so they should not be the
#              thinnest rows in the table.
#
# Capacity: every arm below is compared against a control with more parameters.
#   trilinear+LN+zshared 18,260 | attention+LN+zshared 18,695
#   controls: single ch17 19,161 (stronger on Kvasir) / single+LN ch17 19,195
mkdir -p results/logs

OX="--dataset oxford_pet --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
KV="--dataset kvasir --size 96 96 --batch-size 64 --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"

run() {
    local script="$1" name="$2"; shift 2
    echo "=== $(date '+%H:%M:%S') starting $name ==="
    micromamba run -n ml_env python3 "experiments/${script}" "$@" \
        > "results/logs/iter9_${name}.log" 2>&1 \
        || echo "!!! $name FAILED (see results/logs/iter9_${name}.log)"
    echo "=== $(date '+%H:%M:%S') finished $name ==="
}

# --- A. the cheap-z lead ----------------------------------------------------
run run_seg.py kv_pyr_ln_zshared_s34  $KV --head pyramidal --level-norm --z-shared --seeds 3 4
run run_seg.py kv_attn_ln_zshared     $KV --head pyramidal_attention --level-norm --z-shared --seeds 0 1 2
run run_seg.py ox_pyr_ln_zshared      $OX --head pyramidal --level-norm --z-shared --seeds 0 1 2

# --- B. bring the controls the new verdicts depend on to 5 seeds ------------
run run_seg.py kv_single_ln_ch17_s34  $KV --head pyramidal_single_level --level-norm --channels 17 --seeds 3 4
run run_seg.py kv_reduced_single_ln_ch17_s34 $KV --head pyramidal_reduced_single_level --level-norm --channels 17 --seeds 3 4
run run_seg.py ox_single_ln_ch17_s34  $OX --head pyramidal_single_level --level-norm --channels 17 --seeds 3 4
run run_seg.py ox_single_ln_ch19_s34  $OX --head pyramidal_single_level --level-norm --channels 19 --seeds 3 4

echo "=== iteration-9 batch complete ==="
