#!/bin/bash

set -ex

# cudss license: https://docs.nvidia.com/cuda/cudss/license.html
mkdir tmp_cudss && cd tmp_cudss

arch_path='sbsa'
CUDA_MAJOR="${CUDA_VERSION%%.*}"
export TARGETARCH=${TARGETARCH:-$(uname -m)}
if [ ${TARGETARCH} = 'amd64' ] || [ "${TARGETARCH}" = 'x86_64' ]; then
    arch_path='x86_64'
fi
CUDSS_NAME="libcudss-linux-${arch_path}-0.7.0.20_cuda${CUDA_MAJOR}-archive"
curl --retry 3 -OLs https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-${arch_path}/${CUDSS_NAME}.tar.xz

tar xf ${CUDSS_NAME}.tar.xz
cp -a ${CUDSS_NAME}/include/* /usr/local/cuda/include/
cp -a ${CUDSS_NAME}/lib/* /usr/local/cuda/lib64/

cd ..
rm -rf tmp_cudss
ldconfig
