#!/usr/bin/env bash

set -eou pipefail

TOPDIR=$(git rev-parse --show-toplevel)

for CUDA_VERSION in 12.4 12.1 11.8 cpu; do
  CUDA_VERSION="${CUDA_VERSION}" conda/build_docker.sh
done
