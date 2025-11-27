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

# Detect CUDA 13 from build environment and use appropriate wheel index
if [[ "${BUILD_ENVIRONMENT}" == *cuda13* ]]; then
  CUDA_INDEX_URL="https://download.pytorch.org/whl/cu130"
  echo "BUILD_ENVIRONMENT contains cuda13, using cu130 wheels"
else
  CUDA_INDEX_URL="https://download.pytorch.org/whl/cu128"
  echo "Using stable CUDA 12.8 wheels"
fi

# Stable packages are ok here, just to satisfy TorchBench check
pip_install torch torchvision torchaudio --index-url "${CUDA_INDEX_URL}"

install_torchbench
install_huggingface
install_timm

# Clean up
conda_run pip uninstall -y torch torchvision torchaudio triton torchao
