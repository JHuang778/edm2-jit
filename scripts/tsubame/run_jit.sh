#!/bin/bash
# TSUBAME 4.0 / Altair Grid Engine job template for one MP-JiT training run.
# Submitted via `qsub -g <GROUP> -v CONFIG=<path> run_jit.sh`.
#
# Resource defaults: one full H100 node (4 GPUs) for 24h. Long trainings
# (e.g. 400ep B/16) are covered via chained submission with -hold_jid — see
# submit_matrix.sh. Jobs are idempotent because main_jit.py resumes from
# checkpoint-last.pth automatically when --resume points at the output dir.
#
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o logs/

set -euo pipefail
cd "${SGE_O_WORKDIR:-$PWD}"
mkdir -p logs

# --- environment ---
# shellcheck disable=SC1091
source "$PWD/scripts/tsubame/env.sh"

# --- config ---
: "${CONFIG:?must be passed via -v CONFIG=<path/to/config.sh>}"
# shellcheck disable=SC1090
source "$CONFIG"

: "${RUN_ID:?config must set RUN_ID}"
: "${MODEL:?config must set MODEL}"
: "${EPOCHS:?config must set EPOCHS}"
: "${SEED:=0}"
: "${BATCH_SIZE:=64}"
: "${IMG_SIZE:=256}"
: "${EXTRA_ARGS:=}"

OUTPUT_DIR="${OUTPUT_DIR:-output/$RUN_ID}"
mkdir -p "$OUTPUT_DIR"
RESUME_FLAG=""
if [ -f "$OUTPUT_DIR/checkpoint-last.pth" ]; then
    RESUME_FLAG="--resume $OUTPUT_DIR"
fi

NGPUS="${NGPUS:-4}"
MASTER_PORT="${MASTER_PORT:-$((12000 + RANDOM % 1000))}"

echo "=== $(date -Is)  run_id=$RUN_ID  job_id=${JOB_ID:-local}  host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true

torchrun \
    --standalone \
    --nproc_per_node="$NGPUS" \
    --master_port="$MASTER_PORT" \
    main_jit.py \
    --model "$MODEL" \
    --img_size "$IMG_SIZE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    $RESUME_FLAG \
    $EXTRA_ARGS

echo "=== $(date -Is)  run_id=$RUN_ID  FINISHED ==="
