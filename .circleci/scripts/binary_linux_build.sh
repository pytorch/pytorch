#!/bin/bash

GIT_ROOT_DIR=$(git rev-parse --show-toplevel)

echo "RUNNING ON $(uname -a) WITH $(nproc) CPUS AND $(free -m)"
set -eux -o pipefail
source /env

# Defaults here so they can be changed in one place
# This script is run inside Docker.2XLarge+ container that has 20 CPU cores
# But ncpu will return total number of cores on the system
export MAX_JOBS=18

USE_CUDA=${USE_CUDA:-}
if [[ "${DESIRED_CUDA}" == *"cpu"* ]]; then
  USE_CUDA="1"
fi

# Parse the parameters
if [[ "$PACKAGE_TYPE" == 'conda' ]]; then
  build_script='/builder/conda/build_pytorch.sh'
elif [[ "$DESIRED_CUDA" == *"rocm"* ]]; then
  build_script='/builder/manywheel/build_rocm.sh'
else
  build_script="USE_CUDA='${USE_CUDA}' ${GIT_ROOT_DIR}/packaging/wheel/build.sh"
fi

# Build the package
SKIP_ALL_TESTS=1 stdbuf -i0 -o0 -e0 "$build_script"
