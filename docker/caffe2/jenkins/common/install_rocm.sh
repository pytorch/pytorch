#!/bin/bash

set -ex

install_ubuntu() {
    apt-get update
    apt-get install -y wget

    # AMD's official BLAS library is BLIS (https://github.com/flame/blis) from UT Austin's FLAME group)
    wget https://github.com/flame/blis/archive/0.6.0.tar.gz
    tar xzf 0.6.0.tar.gz
    pushd blis-0.6.0
    ./configure --enable-blas --enable-shared --enable-static -t openmp x86_64
    make -j
    make install
    popd

    # we need an accompanying LAPACK
    wget https://github.com/flame/libflame/archive/5.2.0.tar.gz
    tar xzf 5.2.0.tar.gz
    pushd libflame-5.2.0
    ./configure --enable-dynamic-build --enable-lapack2flame --enable-max-arg-list-hack --enable-supermatrix --disable-ldim-alignment --enable-multithreading=openmp --disable-autodetect-f77-ldflags --disable-autodetect-f77-name-mangling
    make -j
    make install
    popd

    # Need the libc++1 and libc++abi1 libraries to allow torch._C to load at runtime
    apt-get install -y libc++1
    apt-get install -y libc++abi1

    DEB_ROCM_REPO=http://repo.radeon.com/rocm/apt/debian
    # Add rocm repository
    wget -qO - $DEB_ROCM_REPO/rocm.gpg.key | apt-key add -
    echo "deb [arch=amd64] $DEB_ROCM_REPO xenial main" > /etc/apt/sources.list.d/rocm.list
    apt-get update --allow-insecure-repositories

    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
                   rocm-dev \
                   rocm-utils \
                   rocfft \
                   miopen-hip \
                   rocblas \
                   hipsparse \
                   rocrand \
                   hipcub \
                   rocthrust \
                   rccl \
                   rocprofiler-dev \
                   roctracer-dev
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
                   rocm-utils \
                   rocfft \
                   miopen-hip \
                   rocblas \
                   hipsparse \
                   rocrand \
                   rccl \
                   hipcub \
                   rocthrust \
                   rocprofiler-dev \
                   roctracer-dev
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
