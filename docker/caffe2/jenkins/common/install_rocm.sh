#!/bin/bash

set -ex

[ -n "$ROCM_VERSION" ]

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
                   cxlactivitylogger

    export PATH="/opt/rocm/bin:/opt/rocm/hcc/bin:/opt/rocm/hip/bin:/opt/rocm/opencl/bin:${PATH}"
    export MIOPEN_DISABLE_CACHE=1
}

install_centos() {
    echo "Not implemented yet"
    exit 1
}

install_hip_thrust() {
    # Needed for now, will be replaced soon
    git clone --recursive https://github.com/ROCmSoftwarePlatform/Thrust.git /data/Thrust
    rm -rf /data/Thrust/thrust/system/cuda/detail/cub-hip
    git clone --recursive https://github.com/ROCmSoftwarePlatform/cub-hip.git /data/Thrust/thrust/system/cuda/detail/cub-hip
    cd /data/Thrust/thrust/system/cuda/detail/cub-hip && git checkout hip_port_1.7.4_caffe2 && cd -
    export THRUST_ROOT=/data/Thrust
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

install_hip_thrust
