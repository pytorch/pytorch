#!/bin/bash

set -ex

[ -n "$UBUNTU_VERSION" ]
[ -n "$CUDA_VERSION" ]

# The NCCL version is not encoded in the build environment.
# This file installs the latest version that works.

# There are only NCCL packages for Ubuntu 16.04 and 14.04
if [[ "$UBUNTU_VERSION" == 16.04 ]]; then
  NCCL_UBUNTU_VER=ubuntu1604
  NCCL_DEB='nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb'
elif [[ "$UBUNTU_VERSION" == 14.04 ]]; then
  NCCL_UBUNTU_VER=ubuntu1404
  NCCL_DEB='nvidia-machine-learning-repo-ubuntu1404_4.0-2_amd64.deb'
else
  echo "There is no NCCL package for Ubuntu version ${UBUNTU_VERSION}."
  echo "    NCCL will not be installed."
fi

if [ -n "$NCCL_UBUNTU_VER" ]; then

  # The deb is agnostic of CUDA version
  curl -LO "http://developer.download.nvidia.com/compute/machine-learning/repos/${NCCL_UBUNTU_VER}/x86_64/${NCCL_DEB}"

  # This dpkg call needs wget
  apt-get update
  apt-get install -y wget
  dpkg -i "${NCCL_DEB}"

  apt update
  apt install -y libnccl2 libnccl-dev
fi
