#!/bin/bash

echo "RUNNING ON $(uname -a) WITH $(nproc) CPUS AND $(free -m)"
set -eux -o pipefail
source /env

# Because most Circle executors only have 20 CPUs, using more causes OOMs w/ Ninja and nvcc parallelization
MEMORY_LIMIT_MAX_JOBS=18
NUM_CPUS=$(( $(nproc) - 2 ))

# Defaults here for **binary** linux builds so they can be changed in one place
export MAX_JOBS=${MAX_JOBS:-$(( ${NUM_CPUS} > ${MEMORY_LIMIT_MAX_JOBS} ? ${MEMORY_LIMIT_MAX_JOBS} : ${NUM_CPUS} ))}

if [[ "${DESIRED_CUDA}" =~ cu1[1-2][0-9] ]]; then
  export BUILD_SPLIT_CUDA="ON"
fi

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

if [[ "$CIRCLE_BRANCH" == "main" ]] || [[ "$CIRCLE_BRANCH" == "master" ]] || [[ "$CIRCLE_BRANCH" == release/* ]]; then
  export BUILD_DEBUG_INFO=1
fi

# Build the package
SKIP_ALL_TESTS=1 "/builder/$build_script"
