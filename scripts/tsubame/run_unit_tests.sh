#!/bin/bash
# Covers R001 + R002: MPLinear parity + σ-bucket χ² unit tests. Cheap; runs on
# the smallest share. Should finish in <5 min.
#
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=0:30:00
#$ -N jit_unit_tests
#$ -j y
#$ -o logs/

set -euo pipefail
cd "${SGE_O_WORKDIR:-$PWD}"
mkdir -p logs

# shellcheck disable=SC1091
source "$PWD/scripts/tsubame/env.sh"

python tests/test_mplinear.py
python tests/test_sigma_buckets.py
echo "Unit tests passed."
