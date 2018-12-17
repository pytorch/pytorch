#!/bin/bash

set -ex

install_ubuntu() {
    apt-get update
    apt-get install -y wget
    apt-get install -y libopenblas-dev

    # Need the libc++1 and libc++abi1 libraries to allow torch._C to load at runtime
    apt-get install libc++1
    apt-get install libc++abi1

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
                   rocm-profiler \
                   cxlactivitylogger \
                   rocsparse \
                   hipsparse \
                   rocrand \
                   hip-thrust

    pushd /tmp
    wget https://github.com/RadeonOpenCompute/hcc/releases/download/roc-1.9.2-pytorch-eap/hcc-1.2.18473-Linux.deb
    wget https://github.com/ROCm-Developer-Tools/HIP/releases/download/roc-1.9.2-pytorch-eap/hip_base-1.5.18462.deb
    wget https://github.com/ROCm-Developer-Tools/HIP/releases/download/roc-1.9.2-pytorch-eap/hip_doc-1.5.18462.deb
    wget https://github.com/ROCm-Developer-Tools/HIP/releases/download/roc-1.9.2-pytorch-eap/hip_hcc-1.5.18462.deb
    wget https://github.com/ROCm-Developer-Tools/HIP/releases/download/roc-1.9.2-pytorch-eap/hip_samples-1.5.18462.deb
    apt install -y ./hcc-1.2.18473-Linux.deb ./hip_base-1.5.18462.deb ./hip_hcc-1.5.18462.deb ./hip_doc-1.5.18462.deb ./hip_samples-1.5.18462.deb
    popd

    # HIP has a bug that drops DEBUG symbols in generated MakeFiles.
    # https://github.com/ROCm-Developer-Tools/HIP/pull/588
    if [[ -f /opt/rocm/hip/cmake/FindHIP.cmake ]]; then
        sudo sed -i 's/set(_hip_build_configuration "${CMAKE_BUILD_TYPE}")/string(TOUPPER _hip_build_configuration "${CMAKE_BUILD_TYPE}")/' /opt/rocm/hip/cmake/FindHIP.cmake
    fi

    # there is a case-sensitivity issue in the cmake files of some ROCm libraries. Fix this here until the fix is released
    sed -i 's/find_dependency(hip)/find_dependency(HIP)/g' /opt/rocm/rocsparse/lib/cmake/rocsparse/rocsparse-config.cmake
    sed -i 's/find_dependency(hip)/find_dependency(HIP)/g' /opt/rocm/rocfft/lib/cmake/rocfft/rocfft-config.cmake
    sed -i 's/find_dependency(hip)/find_dependency(HIP)/g' /opt/rocm/miopen/lib/cmake/miopen/miopen-config.cmake
    sed -i 's/find_dependency(hip)/find_dependency(HIP)/g' /opt/rocm/rocblas/lib/cmake/rocblas/rocblas-config.cmake
}

install_centos() {

  yum update -y
  yum install -y wget
  yum install -y openblas-devel

  yum install -y epel-release
  yum install -y dkms kernel-headers-`uname -r` kernel-devel-`uname -r`

  echo "[ROCm]" > /etc/yum.repos.d/rocm.repo
  echo "name=ROCm" >> /etc/yum.repos.d/rocm.repo
  echo "baseurl=http://repo.radeon.com/rocm/yum/rpm/" >> /etc/yum.repos.d/rocm.repo
  echo "enabled=1" >> /etc/yum.repos.d/rocm.repo
  echo "gpgcheck=0" >> /etc/yum.repos.d/rocm.repo

  yum update -y

  yum install -y \
                   rocm-dev \
                   rocm-libs \
                   rocm-utils \
                   rocfft \
                   miopen-hip \
                   miopengemm \
                   rocblas \
                   rocm-profiler \
                   cxlactivitylogger \
                   rocsparse \
                   hipsparse \
                   rocrand \
                   rccl

  pushd /tmp
  wget https://github.com/RadeonOpenCompute/hcc/releases/download/roc-1.9.2-pytorch-eap/hcc-1.2.18473-Linux.rpm
  wget https://github.com/ROCm-Developer-Tools/HIP/releases/download/roc-1.9.2-pytorch-eap/hip_base-1.5.18462.rpm
  wget https://github.com/ROCm-Developer-Tools/HIP/releases/download/roc-1.9.2-pytorch-eap/hip_doc-1.5.18462.rpm
  wget https://github.com/ROCm-Developer-Tools/HIP/releases/download/roc-1.9.2-pytorch-eap/hip_hcc-1.5.18462.rpm
  wget https://github.com/ROCm-Developer-Tools/HIP/releases/download/roc-1.9.2-pytorch-eap/hip_samples-1.5.18462.rpm
  rpm -i --replacefiles ./hcc-1.2.18473-Linux.rpm ./hip_base-1.5.18462.rpm ./hip_hcc-1.5.18462.rpm ./hip_doc-1.5.18462.rpm ./hip_samples-1.5.18462.rpm
  popd
  
  # Cleanup
  yum clean all
  rm -rf /var/cache/yum
  rm -rf /var/lib/yum/yumdb
  rm -rf /var/lib/yum/history

  # Needed for now, will be replaced once hip-thrust is packaged for CentOS
  git clone --recursive https://github.com/ROCmSoftwarePlatform/Thrust.git /data/Thrust
  rm -rf /data/Thrust/thrust/system/cuda/detail/cub-hip
  git clone --recursive https://github.com/ROCmSoftwarePlatform/cub-hip.git /data/Thrust/thrust/system/cuda/detail/cub-hip
  ln -s /data/Thrust/thrust /opt/rocm/include/thrust
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
