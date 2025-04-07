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

  # TODO (huydhn): There is no torchvision release on 3.13 when I write this, so
  # I'm using nightly here instead. We just need to package to be able to install
  # TIMM. Removing this once vision has a release on 3.13
  if [[ "${ANACONDA_PYTHON_VERSION}" == "3.13" ]]; then
    pip_install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124
  fi

  pip_install "git+https://github.com/huggingface/pytorch-image-models@${commit}"
  # Clean up
  conda_run pip uninstall -y cmake torch torchvision triton
}

# Pango is needed for weasyprint which is needed for doctr
conda_install pango
install_huggingface
install_timm
