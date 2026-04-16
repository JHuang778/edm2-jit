#!/bin/bash
# Submit G/16 @ 512² scaling pair (R050 + R051). Blocked on M3 gate —
# only launch after the 256² matrix passes its decision gate.
#
# Usage:
#   export TSUBAME_GROUP=tga-xxxxx
#   export N_CHAIN=20      # G/16 @ 512² is ~3× heavier than B/16 @ 256²
#   scripts/tsubame/submit_scaling.sh

set -euo pipefail
: "${TSUBAME_GROUP:?export TSUBAME_GROUP=<your t4 group>}"
N_CHAIN="${N_CHAIN:-20}"
AR="${AR:-}"

cd "$(dirname "$0")/../.."
SCRIPT_DIR="scripts/tsubame"

configs=(
    "$SCRIPT_DIR/configs/R050_Ap_G16_512.sh"
    "$SCRIPT_DIR/configs/R051_Dp_G16_512.sh"
)

for cfg in "${configs[@]}"; do
    name=$(basename "$cfg" .sh)
    prev_jid=""
    for i in $(seq 1 "$N_CHAIN"); do
        job_name="${name}_${i}of${N_CHAIN}"
        hold=""
        ar=""
        [ -n "$prev_jid" ] && hold="-hold_jid $prev_jid"
        [ -n "$AR" ] && ar="-ar $AR"
        jid=$(qsub -terse -g "$TSUBAME_GROUP" -N "$job_name" $hold $ar \
                   -v CONFIG="$PWD/$cfg" "$SCRIPT_DIR/run_jit.sh")
        echo "submitted $job_name  job_id=$jid"
        prev_jid=$jid
    done
done
