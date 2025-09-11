#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

clone_executorch() {
  EXECUTORCH_PINNED_COMMIT=$(get_pinned_commit executorch)

  # Clone the Executorch
  git clone https://github.com/pytorch/executorch.git

  # and fetch the target commit
  pushd executorch
  git checkout "${EXECUTORCH_PINNED_COMMIT}"
  git submodule update --init --recursive
  popd

  chown -R jenkins executorch
}

install_buck2() {
  pushd executorch/.ci/docker

  BUCK2_VERSION=$(cat ci_commit_pins/buck2.txt)
  source common/install_buck.sh

  popd
}

install_conda_dependencies() {
  pushd executorch/.ci/docker
  # Install conda dependencies like flatbuffer
  conda_install --file conda-env-ci.txt
  popd
}

install_pip_dependencies() {
  pushd executorch
  as_jenkins bash install_executorch.sh

  # A workaround, ExecuTorch has moved to numpy 2.0 which is not compatible with the current
  # numba and scipy version used in PyTorch CI
  conda_run pip uninstall -y numba scipy

  popd
}

setup_executorch() {
  pushd executorch

  export PYTHON_EXECUTABLE=python
  export CMAKE_ARGS="-DEXECUTORCH_BUILD_PYBIND=ON -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON"

  as_jenkins .ci/scripts/setup-linux.sh --build-tool cmake || true
  popd
}

clone_executorch
install_buck2
install_conda_dependencies
install_pip_dependencies
setup_executorch
