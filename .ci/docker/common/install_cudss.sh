#!/bin/bash

set -ex

# cudss license: https://docs.nvidia.com/cuda/cudss/license.html
mkdir tmp_cudss && cd tmp_cudss

arch_path='sbsa'
export TARGETARCH=${TARGETARCH:-$(uname -m)}
if [ "${TARGETARCH}" = 'amd64' ] || [ "${TARGETARCH}" = 'x86_64' ]; then
    arch_path='x86_64'
fi

if [[ ${CUDA_VERSION:0:2} == "13" ]]; then
    CUDSS_NAME="libcudss-linux-${arch_path}-0.7.1.4_cuda13-archive"
elif [[ ${CUDA_VERSION:0:4} =~ ^12\.[1-9]$ ]]; then
    CUDSS_NAME="libcudss-linux-${arch_path}-0.7.1.4_cuda12-archive"
else
    CUDSS_NAME=""
fi

if [ -n "${CUDSS_NAME}" ]; then
    curl --retry 3 -OLs https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-${arch_path}/${CUDSS_NAME}.tar.xz

    tar xf ${CUDSS_NAME}.tar.xz
    cp -a ${CUDSS_NAME}/include/* /usr/local/cuda/include/
    cp -a ${CUDSS_NAME}/lib/* /usr/local/cuda/lib64/
fi

cd ..
rm -rf tmp_cudss
ldconfig
