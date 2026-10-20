#!/usr/bin/env bash
# Iteration-14 batch (~11 h, sequential, priority-ordered).
#
# Replaces iteration 13 (structured offsets to n=10), which was stopped 35 min
# in: its own header concedes the Oxford half needs n~25/arm, and its best case
# upgrades a peripheral design-rule row. This batch buys the two seed
# extensions RESULTS S8 rates as resolvable, plus the CIFAR-100 long-schedule
# probe motivated by the loss-sig analysis (RESULTS S6): the pyramid's train
# loss is still descending at 150 epochs (slack +3.4%) while the control is
# flat, so the fitting advantage there is a lower bound.
#
#   A (~4 h)   Kvasir attention+LN to n=6 and its single ch19 control to n=6.
#              Resolves the study's one negative headline ("attention loses at
#              23k") at seed level, and grows n for the one *fitting* deficit.
#   B (~4 h)   Oxford full-z+LN and single+LN ch17 to n=8. At n=8 this is
#              either the first seed-significant val parity win on
#              segmentation, or an honest null (S8: needs 8/arm).
#   C (~3.5 h) CIFAR-100 pyramidal + single at 300 epochs, seeds 0-2. Tagged
#              _ep=300 (run_cls.py change 2026-08-04) so these rows cannot mix
#              with the locked 150-epoch schedule.
#
# Iteration 13 can be resumed later via run_batch_iter13.sh if the structured-
# offset seed counts are ever worth the CPU-hours again.
mkdir -p results/logs

OX="--dataset oxford_pet --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
KV="--dataset kvasir --size 96 96 --batch-size 64 --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
C100="--dataset cifar100 --epochs 300 --num-workers 6"

run() {
    local script="$1"; local name="$2"; shift 2
    echo "=== $(date '+%H:%M:%S') starting $name ==="
    micromamba run -n ml_env python3 "experiments/${script}" "$@" \
        > "results/logs/iter14_${name}.log" 2>&1 \
        || echo "!!! $name FAILED (see results/logs/iter14_${name}.log)"
    echo "=== $(date '+%H:%M:%S') finished $name ==="
}

# --- A. Kvasir ~23k bracket to n=6 per arm -----------------------------------
run run_seg.py kv_attn_ln_s5      $KV --head pyramidal_attention --level-norm --seeds 5
run run_seg.py kv_single_ch19_s35 $KV --head pyramidal_single_level --channels 19 --seeds 3 4 5

# --- B. Oxford ~19.3k bracket to n=8 per arm ---------------------------------
run run_seg.py ox_pyr_ln_s57       $OX --head pyramidal --level-norm --seeds 5 6 7
run run_seg.py ox_single_ln17_s57  $OX --head pyramidal_single_level --level-norm --channels 17 --seeds 5 6 7

# --- C. CIFAR-100, 300-epoch schedule ----------------------------------------
run run_cls.py c100_pyr_ep300    $C100 --head pyramidal --seeds 0 1 2
run run_cls.py c100_single_ep300 $C100 --head pyramidal_single_level --seeds 0 1 2

echo "=== iteration-14 batch complete ==="
