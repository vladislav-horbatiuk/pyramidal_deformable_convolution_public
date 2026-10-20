#!/usr/bin/env bash
# Iteration-20 batch: depth stretch-and-clip, preliminary seed 0 (~70 min, CPU).
#
# The depth is z~ = (L-1) * sigmoid(z), so the first and last pyramid levels
# are reachable only as z -> +-inf, where the gradient vanishes. Trained
# cheap-z+LN models keep |z| small (no samples beyond sigmoid 0.05/0.95), but
# that may be the optimiser avoiding saturation rather than the ends being
# useless. --z-stretch 0.1 maps the depth as clamp(1.2*sigmoid(z) - 0.1, 0, 1):
# both ends become exactly reachable at z ~ +-2.4, where sigmoid' ~ 0.08.
#
# Arm: cheap-z + LN + stretch, paired by seed with the existing cheap-z + LN
# runs (same flags otherwise). This batch is seed 0 only, a preliminary look;
# the test proper is seeds 0-4 per dataset, judged on seed-paired Dice and
# training loss, with the share of samples at exactly the end levels as the
# mechanism check.
mkdir -p results/logs

OX="--dataset oxford_pet --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
KV="--dataset kvasir --size 96 96 --batch-size 64 --epochs 60 --eval-every 5 --num-workers 6 --save-models --dump-per-image"
CHEAP="--head pyramidal --level-norm --z-shared"

run() {
    local script="$1"; local name="$2"; shift 2
    echo "=== $(date '+%H:%M:%S') starting $name ==="
    micromamba run -n ml_env python3 "experiments/${script}" "$@" \
        > "results/logs/iter20_${name}.log" 2>&1 \
        || echo "!!! $name FAILED (see results/logs/iter20_${name}.log)"
    echo "=== $(date '+%H:%M:%S') finished $name ==="
}

run run_seg.py kv_zstretch_s0 $KV $CHEAP --z-stretch 0.1 --seeds 0
run run_seg.py ox_zstretch_s0 $OX $CHEAP --z-stretch 0.1 --seeds 0

echo "=== iteration-20 batch complete ==="
