#!/bin/bash
# Generic single-GPU job template for TSUBAME (Altair Grid Engine).
# Runs an arbitrary command on a GPU node with the standard env (CUDA, conda).
#
# Usage (via convenience wrapper):
#   scripts/tsubame/submit_gpu_task.sh "bash scripts/analysis/run_gate_decomposition.sh R020"
#
# Or directly:
#   qsub -g $TSUBAME_GROUP scripts/tsubame/run_gpu_task.sh \
#        "python analysis_fid_decomposition.py --samples ..."
#
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=2:00:00
#$ -j y
#$ -o logs/

set -euo pipefail
cd "${SGE_O_WORKDIR:-$PWD}"
mkdir -p logs

# --- environment ---
# shellcheck disable=SC1091
source "$PWD/scripts/tsubame/env.sh"

# --- command (passed as positional args) ---
CMD="$*"
if [ -z "$CMD" ]; then
    echo "ERROR: no command given. Pass as arguments." >&2
    exit 1
fi
echo "=== $(date -Is)  job_id=${JOB_ID:-local}  host=$(hostname) ==="
echo "CMD: $CMD"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true

eval "$CMD"

echo "=== $(date -Is)  FINISHED ==="
