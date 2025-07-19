#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

clone_vllm() {
  EXECUTORCH_PINNED_COMMIT=$(get_pinned_commit vllm)

  # Clone the VLLM
  git clone https://github.com/vllm-project/vllm.git

  # and fetch the target commit
  pushd executorch
  git checkout "${EXECUTORCH_PINNED_COMMIT}"
  git submodule update --init --recursive
  popd

  chown -R jenkins executorch
}



clone_vllm
