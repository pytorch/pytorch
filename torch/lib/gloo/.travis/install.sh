#!/bin/bash

set -e

sudo apt-get update
sudo apt-get install -y \
    libeigen3-dev \
    libhiredis-dev \
    libibverbs-dev

if [[ "${BUILD_CUDA}" == 'ON' ]]; then
  ################
  # Install CUDA #
  ################

  source /etc/lsb-release

  REPO="ubuntu1404"
  if [ "${DISTRIB_RELEASE}" == "16.04" ]; then
    REPO="ubuntu1604"
  fi

  CUDA_REPO_PKG="cuda-repo-${REPO}_8.0.44-1_amd64.deb"
  CUDA_PKG_VERSION="8-0"
  CUDA_VERSION="8.0"

  wget "http://developer.download.nvidia.com/compute/cuda/repos/${REPO}/x86_64/${CUDA_REPO_PKG}"
  sudo dpkg -i "${CUDA_REPO_PKG}"
  rm -f "${CUDA_REPO_PKG}"
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
      "cuda-core-${CUDA_PKG_VERSION}" \
      "cuda-cudart-dev-${CUDA_PKG_VERSION}" \
      "cuda-driver-dev-${CUDA_PKG_VERSION}" \
      "cuda-nvrtc-dev-${CUDA_PKG_VERSION}"


  # Manually create CUDA symlink
  sudo ln -sf "/usr/local/cuda-${CUDA_VERSION}" /usr/local/cuda
fi
