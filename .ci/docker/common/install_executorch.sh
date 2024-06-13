#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

clone_executorch() {
  EXECUTORCH_PINNED_COMMIT=$(get_pinned_commit executorch)

  # Clone the Executorch
  as_jenkins git clone https://github.com/pytorch/executorch.git

  # and fetch the target commit
  pushd executorch
  as_jenkins git checkout "${EXECUTORCH_PINNED_COMMIT}"
  as_jenkins git submodule update --init
  popd
}

install_buck2() {
  pushd executorch/.ci/docker

  BUCK2_VERSION=$(cat ci_commit_pins/buck2.txt)
  as_jenkins bash common/install_buck.sh

  popd
}

install_conda_dependencies() {
  pushd executorch/.ci/docker
  # Install conda dependencies like flatbuffer
  conda_install --file conda-env-ci.txt
  popd
}

install_pip_dependencies() {
  pushd executorch/.ci/docker
  # Install all Python dependencies
  pip_install -r requirements-ci.txt
  popd
}

setup_executorch() {
  pushd executorch
  # Setup swiftshader and Vulkan SDK which are required to build the Vulkan delegate
  as_jenkins bash .ci/scripts/setup-vulkan-linux-deps.sh

  export PYTHON_EXECUTABLE=python
  export EXECUTORCH_BUILD_PYBIND=ON
  export CMAKE_ARGS="-DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON"

  as_jenkins .ci/scripts/setup-linux.sh cmake
  popd
}

clone_executorch
install_buck2
install_conda_dependencies
install_pip_dependencies
setup_executorch
