#!/bin/bash
# Inner script launched by ~/jit-autograd/execute.sh on the slurm node.
# execute.sh has already: activated conda `jit`, loaded cuda/openmpi/singularity.
#
# Submit with:
#   /usr/bin/sbatch ~/jit-autograd/execute.sh \
#       /home/hzy980512/edm2-jit/scripts/slurm/run_m1_post_hoc_ema.sh
set -euo pipefail

REPO=/home/hzy980512/edm2-jit
cd "$REPO"

CKPT=${CKPT:-/mnt/nfs/Users/hzy980512/jit_archived/output_imagenet_256/checkpoint-last.pth}
OUT_DIR=${OUT_DIR:-$REPO/output/R010_post_hoc_ema}
N_ALPHA=${N_ALPHA:-32}
NUM_IMAGES=${NUM_IMAGES:-5000}
GEN_BSZ=${GEN_BSZ:-64}

mkdir -p "$OUT_DIR" "$REPO/ssd/tmp/$(basename "$OUT_DIR")"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-$((29400 + RANDOM % 1000))}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

echo "=========================================="
echo "M1 / R010 post-hoc EMA α-sweep"
echo "  repo        : $REPO"
echo "  ckpt        : $CKPT"
echo "  output_dir  : $OUT_DIR"
echo "  n_alpha     : $N_ALPHA"
echo "  num_images  : $NUM_IMAGES"
echo "  gen_bsz     : $GEN_BSZ"
echo "  node        : $(hostname)"
echo "  gpus        : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "  master_port : $MASTER_PORT"
echo "=========================================="

torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    tools/post_hoc_ema.py \
        --ckpt "$CKPT" \
        --output_dir "$OUT_DIR" \
        --n_alpha "$N_ALPHA" \
        --num_images "$NUM_IMAGES" \
        --gen_bsz "$GEN_BSZ"

echo "Done. Verdict:"
cat "$OUT_DIR/verdict.txt" 2>/dev/null || echo "  (verdict.txt not written)"
