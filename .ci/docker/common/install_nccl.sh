#!/bin/bash

set -ex

NCCL_VERSION=""
if [[ ${CUDA_VERSION:0:2} == "11" ]]; then
  NCCL_VERSION=$(cat ci_commit_pins/nccl-cu11.txt)
elif [[ ${CUDA_VERSION:0:2} == "12" ]]; then
  NCCL_VERSION=$(cat ci_commit_pins/nccl-cu12.txt)
fi

if [[ -n "${NCCL_VERSION}" ]]; then
  # NCCL license: https://docs.nvidia.com/deeplearning/nccl/#licenses
  # Follow build: https://github.com/NVIDIA/nccl/tree/master?tab=readme-ov-file#build
  git clone -b $NCCL_VERSION --depth 1 https://github.com/NVIDIA/nccl.git
  cd nccl && make -j src.build
  cp -a build/include/* /usr/local/cuda/include/
  cp -a build/lib/* /usr/local/cuda/lib64/
  cd ..
  rm -rf nccl
  ldconfig
fi
