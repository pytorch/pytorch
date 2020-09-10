#!/bin/bash

GIT_ROOT_DIR=$(git rev-parse --show-toplevel)

echo "RUNNING ON $(uname -a) WITH $(nproc) CPUS AND $(free -m)"
set -eux -o pipefail
source /env

# Defaults here so they can be changed in one place
# This script is run inside Docker.2XLarge+ container that has 20 CPU cores
# But ncpu will return total number of cores on the system
export MAX_JOBS=18

USE_CUDA=${USE_CUDA:-1}
if [[ "${DESIRED_CUDA}" == *"cpu"* ]]; then
  USE_CUDA="0"
fi

# Parse the parameters
if [[ "$PACKAGE_TYPE" == 'conda' ]]; then
  build_script='/builder/conda/build_pytorch.sh'
elif [[ "$DESIRED_CUDA" == *"rocm"* ]]; then
  build_script='/builder/manywheel/build_rocm.sh'
else
  if [[ -n "$DESIRED_PYTHON" && "$DESIRED_PYTHON" != cp* ]]; then
      if [[ "$DESIRED_PYTHON" == '2.7mu' ]]; then
        DESIRED_PYTHON='cp27-cp27mu'
      elif [[ "$DESIRED_PYTHON" == '3.8m' ]]; then
        DESIRED_PYTHON='cp38-cp38'
      else
        python_nodot="$(echo $DESIRED_PYTHON | tr -d m.u)"
        DESIRED_PYTHON="cp${python_nodot}-cp${python_nodot}m"
      fi
  fi
  py_majmin="${DESIRED_PYTHON:2:1}.${DESIRED_PYTHON:3:1}"
  pydir="/opt/python/$DESIRED_PYTHON"
  export PATH="$pydir/bin:$PATH"
  git submodule update --init --recursive
  pip install -q -r requirements.txt
  build_script="${GIT_ROOT_DIR}/packaging/wheel/build.sh"
fi

# Build the package
USE_CUDA="${USE_CUDA}" SKIP_ALL_TESTS=1 stdbuf -i0 -o0 -e0 "$build_script"
