#!/bin/bash

set -ex

NCCL_VERSION=""
if [[ ${CUDA_VERSION:0:2} == "11" ]]; then
  NCCL_VERSION=$(cat ci_commit_pins/nccl-cu11.txt)
elif [[ ${CUDA_VERSION:0:2} == "12" ]]; then
  NCCL_VERSION=$(cat ci_commit_pins/nccl-cu12.txt)
elif [[ ${CUDA_VERSION:0:2} == "13" ]]; then
  NCCL_VERSION=$(cat ci_commit_pins/nccl-cu13.txt)
else
  echo "Unexpected CUDA_VERSION ${CUDA_VERSION}"
  exit 1
fi

if [[ -n "${NCCL_VERSION}" ]]; then
  # NCCL license: https://docs.nvidia.com/deeplearning/nccl/#licenses
  # Follow build: https://github.com/NVIDIA/nccl/tree/master?tab=readme-ov-file#build
  git clone -b $NCCL_VERSION --depth 1 https://github.com/NVIDIA/nccl.git
  pushd nccl
  make -j src.build
  cp -a build/include/* /usr/local/cuda/include/
  cp -a build/lib/* /usr/local/cuda/lib64/
  popd
  rm -rf nccl
  ldconfig
fi
