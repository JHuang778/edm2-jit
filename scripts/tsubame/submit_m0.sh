#!/bin/bash
# Submit M0 sanity runs to TSUBAME: unit tests (R001+R002) on node_q, then
# end-to-end 1k-step bring-up (R003) on node_f. No chaining.
#
# Usage:
#   export TSUBAME_GROUP=tga-xxxxx
#   scripts/tsubame/submit_m0.sh

set -euo pipefail
: "${TSUBAME_GROUP:?export TSUBAME_GROUP=<your t4 group>}"

cd "$(dirname "$0")/../.."
SCRIPT_DIR="scripts/tsubame"

# R001 + R002 — unit tests (pytest).
unit_jid=$(qsub -terse -g "$TSUBAME_GROUP" -N "R001_R002_unit" \
                "$SCRIPT_DIR/run_unit_tests.sh")
echo "submitted R001_R002_unit  job_id=$unit_jid"

# R003 — end-to-end 1k-step sanity, only runs if unit tests pass.
e2e_jid=$(qsub -terse -g "$TSUBAME_GROUP" -N "R003_e2e_1k" \
               -hold_jid "$unit_jid" \
               -v CONFIG="$PWD/$SCRIPT_DIR/configs/R003_e2e_1k.sh" \
               "$SCRIPT_DIR/run_jit.sh")
echo "submitted R003_e2e_1k  job_id=$e2e_jid  (holds on $unit_jid)"
