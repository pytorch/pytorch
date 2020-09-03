#!/bin/bash

echo "RUNNING ON $(uname -a) WITH $(nproc) CPUS AND $(free -m)"
set -eux -o pipefail
source /env

# Defaults here so they can be changed in one place
export MAX_JOBS=${MAX_JOBS:-$(nproc --ignore=1)}

# Parse the parameters
if [[ "$PACKAGE_TYPE" == 'conda' ]]; then
  build_script='conda/build_pytorch.sh'
elif [[ "$DESIRED_CUDA" == cpu ]]; then
  build_script='manywheel/build_cpu.sh'
else
  build_script='manywheel/build.sh'
fi

# Build the package
SKIP_ALL_TESTS=1 stdbuf -i0 -o0 -e0 "/builder/$build_script"
