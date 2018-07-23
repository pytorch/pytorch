#!/bin/bash

set -ex

# TODO: This script should install a SPECIFIC ROCM_VERSION, but actually
# it ignores all values of ROCM_VERSION which are not nightly.  Ugh!
[ -n "$ROCM_VERSION" ]

install_hip_nightly() {
    git clone https://github.com/ROCm-Developer-Tools/HIP.git
    pushd HIP
    export HIP_PLATFORM=hcc
    yes | ./install.sh --install
    popd
    rm -rf HIP
}

install_ubuntu() {
    apt-get update
    apt-get install -y wget

    DEB_ROCM_REPO=http://repo.radeon.com/rocm/apt/debian
    # Add rocm repository
    wget -qO - $DEB_ROCM_REPO/rocm.gpg.key | apt-key add -
    echo "deb [arch=amd64] $DEB_ROCM_REPO xenial main" > /etc/apt/sources.list.d/rocm.list
    apt-get update --allow-insecure-repositories

    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
                   rocm-dev \
                   rocm-libs \
                   rocm-utils \
                   rocfft \
                   miopen-hip \
                   miopengemm \
                   rocblas \
                   hipblas \
                   rocrand \
                   rocm-profiler \
                   hcsparse \
                   cxlactivitylogger
}

install_centos() {
    echo "Not implemented yet"
    exit 1
}

# Install Python packages depending on the base OS
if [ -f /etc/lsb-release ]; then
  install_ubuntu
elif [ -f /etc/os-release ]; then
  install_centos
else
  echo "Unable to determine OS..."
  exit 1
fi

# NB: We first install the "wrong" version, but then use those dev tools
# to install the newer version of HIP.
if [ "$ROCM_VERSION" = "nightly" ]; then
  install_hip_nightly
fi

install_hip_thrust
