#!/bin/bash
# Stage-gate signal #1: per-epoch FID mean-term / cov-term decomposition for
# the Phase 0b pilot (R020) and main-matrix ep-100 checkpoints. Reads EMA
# sample dumps preserved by evaluate() when --keep_gate_samples is active; no
# extra training cost. Produces a CSV that `experiment_gate.py` reads to decide
# M2→M3.
#
# Usage:
#   scripts/analysis/run_gate_decomposition.sh <OUTPUT_DIR> <EPOCHS...>
#
# OUTPUT_DIR must match what was passed to main_jit.py via --output_dir
# (typically `output/<RUN_ID>` per scripts/tsubame/run_jit.sh).
# Example:
#   scripts/analysis/run_gate_decomposition.sh output/R020_D_gate_pilot 50 60 70 80 90 99
#
# NOTE: training loop is zero-based (epoch ∈ [0, EPOCHS)), so the final
# human-"epoch 100" checkpoint is actually saved/tagged as ep099.

set -euo pipefail
OUT_DIR="${1:?output dir (matches args.output_dir; e.g. output/R020_D_gate_pilot)}"
shift
EPOCHS="${*:-50 60 70 80 90 99}"

# Tolerate users passing the bare RUN_ID (no `output/` prefix) by checking both.
if [ ! -d "ssd/tmp/$OUT_DIR" ] && [ -d "ssd/tmp/output/$OUT_DIR" ]; then
    echo "Note: '$OUT_DIR' not found; using 'output/$OUT_DIR' instead." >&2
    OUT_DIR="output/$OUT_DIR"
fi

cd "$(dirname "$0")/../.."

FID_STATS="fid_stats/jit_in256_stats.npz"
CSV_OUT="ssd/tmp/$OUT_DIR/mean_cov_decomposition.csv"
mkdir -p "$(dirname "$CSV_OUT")"
echo "epoch,mean_term,cov_term,fid_total" > "$CSV_OUT"

for ep in $EPOCHS; do
    ep_pad=$(printf "%03d" "$ep")
    SAMPLE_DIR=$(ls -d "ssd/tmp/$OUT_DIR/ep${ep_pad}"/ema-*image5000*-res256 2>/dev/null | head -1 || true)
    if [[ -z "$SAMPLE_DIR" ]]; then
        echo "WARNING: no ema sample dir for ep=$ep at ssd/tmp/$OUT_DIR/ep${ep_pad}/; " \
             "did training pass --online_eval --keep_gate_samples?" >&2
        continue
    fi
    python analysis_fid_decomposition.py \
        --samples "$SAMPLE_DIR" \
        --fid_stats "$FID_STATS" \
        --epoch "$ep" \
        --append-csv "$CSV_OUT"
done

echo "Decomposition written to $CSV_OUT"
echo "Next: experiment_gate.py reads the slope of mean_term over ep 50..100."
