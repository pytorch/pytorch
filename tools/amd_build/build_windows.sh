#!/bin/bash

# Thin wrapper around .ci/pytorch/win-build.sh for testing on a dev machine.
# This script is hacky... starting with it for bootstrapping then may delete.

set -ex -o pipefail

if [ ! -f setup.py ]; then
  echo "ERROR: Please run this build script from PyTorch root directory."
  exit 1
fi

# TODO: should this be "win-rocm-" or "windows-rocm-"?
#   .ci/pytorch/common-build.sh looks for "*win-*" when configuring sccache
#   some workflows use "windows-binary-wheel" or "windows-arm64-binary-wheel"
#   other workflows use "win-vs2022-cpu-py3", "win=vs2022-cuda12.6-py3", etc.
export BUILD_ENVIRONMENT=windows-rocm-manywheel
export PYTORCH_ROCM_ARCH=gfx1100

# To test what CI does (may require software installed at certain paths on your system)
function ci_build() {
  bash .ci/pytorch/win-build.sh
}

# To test without other CI setup.
function dev_build() {
  source .ci/pytorch/common.sh
  source .ci/pytorch/common-build.sh
  source .ci/pytorch/rocm_sdk-build.sh
  python tools/amd_build/build_amd.py
  python tools/amd_build/write_rocm_init.py

  python setup.py bdist_wheel
}

# Toggle comments here to change script modes
# TODO: better ergonomics once we figure out CI and dev workflows.
#       we used a Python build script downstream that could be attractive
#       here as well:
#       https://github.com/ROCm/TheRock/blob/main/external-builds/pytorch/build_prod_wheels.py

# ci_build
dev_build
