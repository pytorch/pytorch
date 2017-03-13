#!/bin/bash
set -e

sudo apt-get update
sudo apt-get install \
    libeigen3-dev \
    libgtest-dev \
    libhiredis-dev \
    libibverbs-dev

if [[ $BUILD_CUDA == 'ON' ]]; then
  
  ################
  # Install CUDA #
  ################

  CUDA_REPO_PKG="cuda-repo-ubuntu1404_8.0.44-1_amd64.deb"
  CUDA_PKG_VERSION="8-0"
  CUDA_VERSION="8.0"

  wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/$CUDA_REPO_PKG
  sudo dpkg -i $CUDA_REPO_PKG
  rm $CUDA_REPO_PKG
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
      cuda-core-$CUDA_PKG_VERSION \
      cuda-driver-dev-$CUDA_PKG_VERSION \

  # manually create CUDA symlink
  sudo ln -s /usr/local/cuda-$CUDA_VERSION /usr/local/cuda
fi