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

    pushd $HOME
    # install hcrng
    curl https://s3.amazonaws.com/ossci-linux/hcrng-master-a8c6a0b-Linux.deb -o hcrng.deb
    dpkg -i hcrng.deb
    rm hcrng.deb

    # install hcsparse
    curl https://s3.amazonaws.com/ossci-linux/hcsparse-master-907a505-Linux.deb -o hcsparse.deb
    dpkg -i hcsparse.deb
    rm hcsparse.deb

    # hotfix a bug in hip's cmake files, this has been fixed in
    # https://github.com/ROCm-Developer-Tools/HIP/pull/516 but for
    # some reason it has not included in the latest rocm release
    if [[ -f /opt/rocm/hip/cmake/FindHIP.cmake ]]; then
        sudo sed -i 's/\ -I${dir}/\ $<$<BOOL:${dir}>:-I${dir}>/' /opt/rocm/hip/cmake/FindHIP.cmake
    fi
}

install_centos() {
    echo "Not implemented yet"
    exit 1
}
  
# This will be removed after merging an upcoming PR.
install_hcrng() {
    mkdir -p /opt/rocm/debians
    curl https://s3.amazonaws.com/ossci-linux/hcrng-master-a8c6a0b-Linux.deb -o /opt/rocm/debians/hcrng.deb 
    dpkg -i /opt/rocm/debians/hcrng.deb
}

install_hcsparse() {	
    mkdir -p /opt/rocm/debians	
    curl https://s3.amazonaws.com/ossci-linux/hcsparse-master-907a505-Linux.deb -o /opt/rocm/debians/hcsparse.deb	
    dpkg -i /opt/rocm/debians/hcsparse.deb	
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

install_hcrng
install_hcsparse
