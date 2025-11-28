#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

function install_huggingface() {
  pip_install -r huggingface-requirements.txt
}

function install_timm() {
  local commit
  commit=$(get_pinned_commit timm)

  pip_install "git+https://github.com/huggingface/pytorch-image-models@${commit}"
}

function install_torchbench() {
  local commit
  commit=$(get_pinned_commit torchbench)
  git clone https://github.com/pytorch/benchmark torchbench
  pushd torchbench
  git checkout "$commit"

  python install.py --continue_on_fail

  echo "Print all dependencies after TorchBench is installed"
  python -mpip freeze
  popd

  chown -R jenkins torchbench
  chown -R jenkins /opt/conda
}

# Pango is needed for weasyprint which is needed for doctr
conda_install pango

# Detect CUDA version and use appropriate wheel index
# DESIRED_CUDA is set as ENV in the Dockerfile (e.g., "13.0.2", "12.8.1")
if [[ "${DESIRED_CUDA}" == 13.* ]]; then
  CUDA_INDEX_URL="https://download.pytorch.org/whl/cu130"
  echo "DESIRED_CUDA=${DESIRED_CUDA}, using cu130 wheels"
else
  # Default to cu128 for CUDA 12.x
  CUDA_INDEX_URL="https://download.pytorch.org/whl/cu128"
  echo "DESIRED_CUDA=${DESIRED_CUDA}, using cu128 wheels"
fi

# Stable packages are ok here, just to satisfy TorchBench check
pip_install torch torchvision torchaudio --index-url "${CUDA_INDEX_URL}"

install_torchbench
install_huggingface
install_timm

# Clean up
conda_run pip uninstall -y torch torchvision torchaudio triton torchao
