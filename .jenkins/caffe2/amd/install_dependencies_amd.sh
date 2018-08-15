#!/bin/bash

set -ex

apt-get update
apt-get install -y --no-install-recommends \
        autoconf \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libgoogle-glog-dev \
        libhiredis-dev \
        libiomp-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libpthread-stubs0-dev \
        libsnappy-dev \
        libprotobuf-dev \
        protobuf-compiler \
        python-dev \
        python-setuptools \
        python-pip

pip install networkx==2.0

pip install --user \
    click \
    future \
    hypothesis \
    jupyter \
    numpy \
    protobuf \
    pytest \
    pyyaml \
    scipy==0.19.1 \
    scikit-image \
    tabulate \
    virtualenv \
    mock \
    typing \
    typing-extensions \
    pyyaml

CLANG_VERSION=3.8

apt-get update
apt-get install -y --no-install-recommends clang-"$CLANG_VERSION"
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Use update-alternatives to make this version the default
update-alternatives --install /usr/bin/gcc gcc /usr/bin/clang-"$CLANG_VERSION" 50
update-alternatives --install /usr/bin/g++ g++ /usr/bin/clang++-"$CLANG_VERSION" 50


apt-get install -y --no-install-recommends \
        rocm-dev \
        rocm-utils \
        rocm-libs \
        rocfft \
        miopen-hip \
        miopengemm \
        rocblas \
        rocrand 


if [[ -f /opt/rocm/hip/cmake/FindHIP.cmake ]]; then
    sudo sed -i 's/\ -I${dir}/\ $<$<BOOL:${dir}>:-I${dir}>/' /opt/rocm/hip/cmake/FindHIP.cmake
fi

if [[ -f /opt/rocm/hip/cmake/FindHIP.cmake ]]; then
    sudo sed -i 's/set(_hip_build_configuration "${CMAKE_BUILD_TYPE}")/string(TOUPPER _hip_build_configuration "${CMAKE_BUILD_TYPE}")/' /opt/rocm/hip/cmake/FindHIP.cmake
fi

git clone --recursive https://github.com/ROCmSoftwarePlatform/Thrust.git /data/Thrust
rm -rf /data/Thrust/thrust/system/cuda/detail/cub-hip
git clone --recursive https://github.com/ROCmSoftwarePlatform/cub-hip.git /data/Thrust/thrust/system/cuda/detail/cub-hip


export PATH=/opt/rocm/bin:$PATH
export PATH=/opt/rocm/hcc/bin:$PATH
export PATH=/opt/rocm/hip/bin:$PATH
export PATH=/opt/rocm/opencl/bin:$PATH
export THRUST_ROOT=/data/Thrust
export HIP_PLATFORM=hcc
