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
  if command -v conda >/dev/null 2>&1; then
    # Install conda dependencies like flatbuffer
    conda_install --file conda-env-ci.txt
  else
    # venv image: install the conda-env-ci.txt equivalents from apt + pip
    sudo apt-get update
    sudo apt-get install -y libuv1-dev libomp-dev pkg-config
    pip_install cmake==3.31.2 ninja==1.10.2
  fi
  popd
}

install_pip_dependencies() {
  pushd executorch
  as_jenkins bash install_executorch.sh

  # A workaround, ExecuTorch has moved to numpy 2.0 which is not compatible with the current
  # numba and scipy version used in PyTorch CI
  env_run pip uninstall -y numba scipy
  # Yaspin is needed for running CI test (get_benchmark_analysis_data.py)
  pip_install yaspin==3.1.0

  popd
}

setup_executorch() {
  export PYTHON_EXECUTABLE=python
  export CMAKE_ARGS="-DEXECUTORCH_BUILD_PYBIND=ON -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON -DEXECUTORCH_BUILD_TESTS=ON"

  as_jenkins .ci/scripts/setup-linux.sh --build-tool cmake || true
}

if [ $# -eq 0 ]; then
  clone_executorch
  install_buck2
  install_conda_dependencies
  install_pip_dependencies
  pushd executorch
  setup_executorch
  popd
else
  "$@"
fi
