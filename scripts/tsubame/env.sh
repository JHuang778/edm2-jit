# Sourced by every TSUBAME job. Customize for your account.
# Docs: https://www.t4.cii.isct.ac.jp/docs/faq.en/scheduler/
set -euo pipefail

# Modules — load CUDA/OpenMPI as available on T4. `|| true` so the job does not
# fail when a module name is renamed; the bundled conda env provides the
# matching runtime anyway.
source /etc/profile.d/modules.sh
module purge
module load cuda/12.6.0 || true
module load openmpi/5.0.2-gcc || true

# Conda env with PyTorch + torch_fidelity matching the vanilla JiT env.
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jit

# Dataset lives on the group fast-scratch filesystem (observed in git log of
# the vanilla repo: "Fix dataset path to /gs/bs/hp190122/jiang/dataset").
export DATA_PATH="${DATA_PATH:-/gs/bs/hp190122/jiang/dataset}"
export FID_STATS_DIR="${FID_STATS_DIR:-$PWD/fid_stats}"

# Help NCCL on H100 node-local all-reduce.
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export OMP_NUM_THREADS=8

# For bf16 matmul paths.
export TORCH_CUDNN_V8_API_ENABLED=1
