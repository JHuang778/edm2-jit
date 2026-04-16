#!/bin/bash
# Submit the main 4-cell A/B/C/D × 3-seed matrix (R030..R037) to TSUBAME.
#
# Each config is submitted as a chain of $N_CHAIN jobs with -hold_jid so that
# 400-epoch trainings can span multiple 24h walltime slots. main_jit.py
# auto-resumes from checkpoint-last.pth, so the chain is idempotent — if a run
# finishes early the subsequent links exit immediately.
#
# Usage:
#   export TSUBAME_GROUP=tga-xxxxx       # your T4 group code
#   export N_CHAIN=14                    # 14 × 24h ≈ 336h upper bound per run
#   scripts/tsubame/submit_matrix.sh
#
# To submit a subset, pass config basenames as arguments:
#   scripts/tsubame/submit_matrix.sh R035_D_full_seed1 R036_D_full_seed2

set -euo pipefail
: "${TSUBAME_GROUP:?export TSUBAME_GROUP=<your t4 group, e.g. tga-xxxxx>}"
N_CHAIN="${N_CHAIN:-14}"
AR="${AR:-}"

cd "$(dirname "$0")/../.."   # repo root
SCRIPT_DIR="scripts/tsubame"

if [ $# -gt 0 ]; then
    configs=()
    for arg in "$@"; do
        path="$SCRIPT_DIR/configs/${arg}.sh"
        [ -f "$path" ] || { echo "missing config: $path" >&2; exit 1; }
        configs+=("$path")
    done
else
    configs=(
        "$SCRIPT_DIR/configs/R030_A_seed1.sh"
        "$SCRIPT_DIR/configs/R031_A_seed2.sh"
        "$SCRIPT_DIR/configs/R032_A_seed3.sh"
        "$SCRIPT_DIR/configs/R033_B_mp_only.sh"
        "$SCRIPT_DIR/configs/R034_C_sigma_only.sh"
        "$SCRIPT_DIR/configs/R035_D_full_seed1.sh"
        "$SCRIPT_DIR/configs/R036_D_full_seed2.sh"
        "$SCRIPT_DIR/configs/R037_D_full_seed3.sh"
    )
fi

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
        echo "submitted $job_name  job_id=$jid  config=$cfg"
        prev_jid=$jid
    done
done
