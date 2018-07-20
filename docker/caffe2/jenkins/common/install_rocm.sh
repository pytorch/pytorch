#!/bin/bash

set -ex

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

    mkdir -p /opt/rocm/debians
    curl https://s3.amazonaws.com/ossci-linux/hcrng-master-a8c6a0b-Linux.deb -o /opt/rocm/debians/hcrng.deb
    dpkg -i /opt/rocm/debians/hcrng.deb

    mkdir -p /opt/rocm/debians
    curl https://s3.amazonaws.com/ossci-linux/hcsparse-master-907a505-Linux.deb -o /opt/rocm/debians/hcsparse.deb
    dpkg -i /opt/rocm/debians/hcsparse.deb
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
