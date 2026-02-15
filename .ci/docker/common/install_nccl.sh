#!/bin/bash

set -ex

# Most of the time, NCCL version won't diverge for different CUDA versions,
# so we can just use the default NCCL version.
NCCL_VERSION=$(cat ci_commit_pins/nccl.txt)

# If NCCL version diverges for different CUDA versions, uncomment the following
# block and add the appropriate files (using CUDA 11 and 12 as an example)

# if [[ ${CUDA_VERSION:0:2} == "11" ]]; then
#   NCCL_VERSION=$(cat ci_commit_pins/nccl-cu11.txt)
# elif [[ ${CUDA_VERSION:0:2} == "12" ]]; then
#   NCCL_VERSION=$(cat ci_commit_pins/nccl-cu12.txt)
# else
#   echo "Unexpected CUDA_VERSION ${CUDA_VERSION}"
#   exit 1
# fi

if [[ -n "${NCCL_VERSION}" ]]; then
  # NCCL license: https://docs.nvidia.com/deeplearning/nccl/#licenses
  # Follow build: https://github.com/NVIDIA/nccl/tree/master?tab=readme-ov-file#build
  git clone -b $NCCL_VERSION --depth 1 https://github.com/NVIDIA/nccl.git
  pushd nccl
  NCCL_MAKE_ARGS=(-j src.build)
  if command -v sccache &>/dev/null; then
    NCCL_MAKE_ARGS+=(NVCC="sccache nvcc")
  fi
  make "${NCCL_MAKE_ARGS[@]}"
  cp -a build/include/* /usr/local/cuda/include/
  cp -a build/lib/* /usr/local/cuda/lib64/
  popd
  rm -rf nccl
  ldconfig
fi
