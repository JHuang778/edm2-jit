#!/bin/bash
# Submit the Phase 0b stage-gate pilot (R020). 100 epochs of MP-JiT-D at
# B/16 256². Three-signal decision is read at epoch 100 before launching the
# full 400ep matrix.
#
# Usage:
#   export TSUBAME_GROUP=tga-xxxxx
#   export N_CHAIN=4      # 4 × 24h ≈ 96h upper bound for 100ep
#   scripts/tsubame/submit_gate.sh

set -euo pipefail
: "${TSUBAME_GROUP:?export TSUBAME_GROUP=<your t4 group>}"
N_CHAIN="${N_CHAIN:-4}"

cd "$(dirname "$0")/../.."
SCRIPT_DIR="scripts/tsubame"
cfg="$SCRIPT_DIR/configs/R020_gate_pilot.sh"

prev_jid=""
for i in $(seq 1 "$N_CHAIN"); do
    job_name="R020_gate_pilot_${i}of${N_CHAIN}"
    hold=""
    [ -n "$prev_jid" ] && hold="-hold_jid $prev_jid"
    jid=$(qsub -terse -g "$TSUBAME_GROUP" -N "$job_name" $hold \
               -v CONFIG="$PWD/$cfg" "$SCRIPT_DIR/run_jit.sh")
    echo "submitted $job_name  job_id=$jid"
    prev_jid=$jid
done
