#!/bin/bash
set -eux -o pipefail

retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

retry git clone https://github.com/peterjc123/builder.git -b circleci_scripts_windows "/c/b"
cd "/c/b"

configs=($BUILD_ENVIRONMENT)
export PACKAGE_TYPE="${configs[0]}"
export DESIRED_PYTHON="${configs[1]}"
export CUDA_VERSION="${configs[2]/cu/}"
export LIBTORCH_CONFIG="${configs[3]}"
export VC_YEAR=2017
export USE_SCCACHE=1
export SCCACHE_BUCKET=ossci-compiler-cache-circleci-v2

set +x
export AWS_ACCESS_KEY_ID=${CIRCLECI_AWS_ACCESS_KEY_FOR_SCCACHE_S3_BUCKET_V4:-}
export AWS_SECRET_ACCESS_KEY=${CIRCLECI_AWS_SECRET_KEY_FOR_SCCACHE_S3_BUCKET_V4:-}

cat >"$HOME/sccache_init.bat" <<EOL
@echo off
set SCCACHE_BUCKET=${SCCACHE_BUCKET}
set AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
set AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
EOL
set -x

if [[ "$CIRCLECI" == 'true' && -d "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019" ]]; then
  rm -rf "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019"
fi

echo "Free space on filesystem before build:"
df -h

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

echo "Free space on filesystem after build:"
df -h
