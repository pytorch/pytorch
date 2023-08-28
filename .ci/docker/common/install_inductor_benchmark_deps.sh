#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

function install_huggingface() {
  local version
  version=$(get_pinned_commit huggingface)
  pip_install pandas
  pip_install scipy
  pip_install z3-solver
  pip_install "transformers==${version}"
}

function install_timm() {
  local commit
  version=$(get_pinned_commit timm)
  pip_install pandas
  pip_install scipy
  pip_install z3-solver
  pip_install "timm==${version}"
}

# Pango is needed for weasyprint which is needed for doctr
conda_install pango
install_huggingface
install_timm
