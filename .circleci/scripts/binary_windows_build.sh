#!/bin/bash
set -eux -o pipefail

git clone https://github.com/peterjc123/builder.git -b circleci_scripts_windows "/c/b"
cd "/c/b"

configs=($BUILD_ENVIRONMENT)
export PACKAGE_TYPE="${configs[0]}"
export DESIRED_PYTHON="${configs[1]}"
export CUDA_VERSION="${configs[2]/cu/}"
export LIBTORCH_CONFIG="${configs[3]}"
export VC_YEAR=2017
export USE_SCCACHE=1

if [[ "$PACKAGE_TYPE" == 'conda' ]]; then
  ./windows/internal/build_conda.bat
elif [[ "$PACKAGE_TYPE" == 'wheel' ]]; then
  ./windows/internal/build_wheels.bat
elif [[ "$PACKAGE_TYPE" == 'libtorch' ]]; then
  export BUILD_PYTHONLESS=1
  if [[ "$LIBTORCH_CONFIG" == 'debug' ]]; then
    export DEBUG=1
  fi
  ./windows/internal/build_wheels.bat
fi
