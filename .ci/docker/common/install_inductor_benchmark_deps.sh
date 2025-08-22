#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

function install_huggingface() {
  local version
  commit=$(get_pinned_commit huggingface)
  pip_install "git+https://github.com/huggingface/transformers@${commit}"
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

  # soxr comes from https://github.com/huggingface/transformers/pull/39429
  pip install transformers==4.54.0 soxr==0.5.0

  echo "Print all dependencies after TorchBench is installed"
  python -mpip freeze
  popd

  chown -R jenkins torchbench
  chown -R jenkins /opt/conda
}

# Pango is needed for weasyprint which is needed for doctr
conda_install pango

# Stable packages are ok here, just to satisfy TorchBench check
pip_install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

install_torchbench
install_huggingface
install_timm

# Clean up
conda_run pip uninstall -y torch torchvision torchaudio triton torchao
