#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

function install_huggingface() {
  local version
  commit=$(get_pinned_commit huggingface)
  pip_install pandas
  pip_install scipy
  pip_install z3-solver
  pip_install "git+https://github.com/huggingface/transformers@${commit}"
}

function install_timm() {
  local commit
  commit=$(get_pinned_commit timm)
  pip_install pandas
  pip_install scipy
  pip_install z3-solver
  pip_install "git+https://github.com/rwightman/pytorch-image-models@${commit}"
}

# Pango is needed for weasyprint which is needed for doctr
conda_install pango
install_huggingface
# install_timm
