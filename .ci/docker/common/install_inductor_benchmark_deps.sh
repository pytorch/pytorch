#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

function install_huggingface() {
  local version
  version=$(get_pinned_commit huggingface)
  pip_install pandas==2.0.3
  pip_install "transformers==${version}"
}

function install_timm() {
  local commit
  commit=$(get_pinned_commit timm)
  pip_install pandas==2.0.3
  pip_install "git+https://github.com/huggingface/pytorch-image-models@${commit}"
  # Clean up
  conda_run pip uninstall -y cmake torch torchvision triton
}

# Pango is needed for weasyprint which is needed for doctr
conda_install pango
install_huggingface
install_timm
