#!/bin/bash

echo "RUNNING ON $(uname -a) WITH $(nproc) CPUS AND $(free -m)"
set -eux -o pipefail
source /env

# Defaults here so they can be changed in one place
# This script is run inside Docker.2XLarge+ container that has 20 CPU cores
# But ncpu will return total number of cores on the system
export MAX_JOBS=18

# Parse the parameters
if [[ "$PACKAGE_TYPE" == 'conda' ]]; then
  build_script='conda/build_pytorch.sh'
elif [[ "$DESIRED_CUDA" == cpu ]]; then
  build_script='manywheel/build_cpu.sh'
elif [[ "$DESIRED_CUDA" == *"rocm"* ]]; then
  build_script='manywheel/build_rocm.sh'
else
  build_script='manywheel/build.sh'
fi

# Build the package
SKIP_ALL_TESTS=1 "/builder/$build_script"
