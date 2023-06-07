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
  pip_uninstall torch
  pip_uninstall torchvision
}

if [ -n "${CONDA_CMAKE}" ]; then
  # Keep the current cmake and numpy version here, so we can reinstall them later
  CMAKE_VERSION=$(get_conda_version cmake)
  NUMPY_VERSION=$(get_conda_version numpy)
fi

install_huggingface
install_timm

if [ -n "${CONDA_CMAKE}" ]; then
  # TODO: This is to make sure that the same cmake and numpy version from install conda
  # script is used. Without this step, the newer cmake version (3.25.2) downloaded by
  # triton build step via pip will fail to detect conda MKL. Once that issue is fixed,
  # this can be removed.
  #
  # The correct numpy version also needs to be set here because conda claims that it
  # causes inconsistent environment.  Without this, conda will attempt to install the
  # latest numpy version, which fails ASAN tests with the following import error: Numba
  # needs NumPy 1.20 or less.
  conda_reinstall cmake="${CMAKE_VERSION}"
  conda_reinstall numpy="${NUMPY_VERSION}"
fi