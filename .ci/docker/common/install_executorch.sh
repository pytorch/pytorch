#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

clone_executorch() {
  EXECUTORCH_PINNED_COMMIT=$(get_pinned_commit executorch)

  # Clone the Executorch
  git clone https://github.com/pytorch/executorch.git
  chown -R jenkins executorch

  # and fetch the target commit
  pushd executorch
  git checkout "${EXECUTORCH_PINNED_COMMIT}"
  pod
}

setup_executorch() {
  pushd executorch
  # Install all ET dependencies and build executorch
  source .ci/scripts/utils.sh
  as_jenkins install_pip_dependencies

  install_flatc_from_source

  as_jenkins install_executorch
  as_jenkins build_executorch_runner "buck2"
  popd
}

clone_executorch
install_flatc_from_source
setup_executorch
