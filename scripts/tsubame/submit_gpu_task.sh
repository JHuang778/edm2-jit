#!/bin/bash
# Convenience wrapper: submit a one-off GPU command to TSUBAME.
#
# Usage:
#   export TSUBAME_GROUP=tga-xxxxx
#   scripts/tsubame/submit_gpu_task.sh "python analysis_fid_decomposition.py --samples ..."
#   scripts/tsubame/submit_gpu_task.sh "bash scripts/analysis/run_gate_decomposition.sh R020_D_gate_pilot"
#
# Options (env vars):
#   AR=6692              — submit to a reservation
#   JOB_NAME=my_task     — custom job name (default: gpu_task)
#   WALLTIME=4:00:00     — override walltime (default: 2h)
#   PREV_JID=12345       — wait for a previous job to finish first

set -euo pipefail
: "${TSUBAME_GROUP:?export TSUBAME_GROUP=<your t4 group>}"
CMD="${1:?usage: submit_gpu_task.sh '<command>'}"
AR="${AR:-}"
JOB_NAME="${JOB_NAME:-gpu_task}"
PREV_JID="${PREV_JID:-}"

cd "$(dirname "$0")/../.."
SCRIPT_DIR="scripts/tsubame"

ar=""
hold=""
[ -n "$AR" ] && ar="-ar $AR"
[ -n "$PREV_JID" ] && hold="-hold_jid $PREV_JID"

jid=$(qsub -terse -g "$TSUBAME_GROUP" -N "$JOB_NAME" $hold $ar \
           -v CMD="$CMD" "$SCRIPT_DIR/run_gpu_task.sh")
echo "submitted $JOB_NAME  job_id=$jid"
echo "  cmd: $CMD"
echo "  log: logs/${JOB_NAME}.o${jid}"
