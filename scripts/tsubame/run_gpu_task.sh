#!/bin/bash
# Generic single-GPU job template for TSUBAME (Altair Grid Engine).
# Runs an arbitrary command on a GPU node with the standard env (CUDA, conda).
#
# Usage:
#   qsub -g $TSUBAME_GROUP [-ar $AR] \
#        -v CMD="python analysis_fid_decomposition.py --samples ... --fid_stats ..." \
#        scripts/tsubame/run_gpu_task.sh
#
# Or via the convenience wrapper:
#   scripts/tsubame/submit_gpu_task.sh "python analysis_fid_decomposition.py --samples ..."
#
# Defaults: 1 GPU (node_q), 2h walltime. Override via env:
#   WALLTIME=4:00:00   — longer jobs
#   NODE_TYPE=node_f   — full node (4 GPUs)
#
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=2:00:00
#$ -j y
#$ -o logs/

set -euo pipefail
cd "${SGE_O_WORKDIR:-$PWD}"
mkdir -p logs

# --- environment ---
# shellcheck disable=SC1091
source "$PWD/scripts/tsubame/env.sh"

# --- command ---
: "${CMD:?must be passed via -v CMD='<command>'}"
echo "=== $(date -Is)  job_id=${JOB_ID:-local}  host=$(hostname) ==="
echo "CMD: $CMD"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true

eval "$CMD"

echo "=== $(date -Is)  FINISHED ==="
