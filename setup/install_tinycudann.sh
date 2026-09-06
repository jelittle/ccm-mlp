#!/usr/bin/env bash
# Optional: build tiny-cuda-nn for the fused MLP kernels.
#
# You do NOT need this to run the code. The pure-PyTorch network path is the
# default and is numerically equivalent; tinycudann is a speed optimisation and
# is what the paper's timing numbers were measured with.
#
# It is a CUDA source build, not a pip package, so it needs a matching CUDA
# toolkit and the CUDA headers on the include path.
set -euo pipefail

# Point these at your CUDA installation if the build cannot find the headers.
# (The original runs used a conda-provided toolkit; adjust for your system.)
#   export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:${CPLUS_INCLUDE_PATH:-}"
#   export C_INCLUDE_PATH="$CONDA_PREFIX/include:${C_INCLUDE_PATH:-}"

pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

echo
echo "Built. Enable it per-model with:"
echo "  model:"
echo "    use_tinycudann: true"
