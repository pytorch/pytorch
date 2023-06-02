#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

function install_huggingface() {
  local version
  version=$(get_pinned_commit huggingface)
  pip_install pandas
  pip_install scipy
  pip_install "transformers==${version}"
}

function install_timm() {
  local commit
  commit=$(get_pinned_commit timm)
  pip_install pandas
  pip_install scipy
  pip_install "git+https://github.com/rwightman/pytorch-image-models@${commit}"
}

install_huggingface
install_timm