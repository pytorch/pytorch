#!/bin/bash

set -ex

# cuSPARSELt license: https://docs.nvidia.com/cuda/cusparselt/license.html
mkdir tmp_cusparselt && cd tmp_cusparselt

cusparselt_version=$1

arch_path='sbsa'
export TARGETARCH=${TARGETARCH:-$(uname -m)}
if [ ${TARGETARCH} = 'amd64' ] || [ "${TARGETARCH}" = 'x86_64' ]; then
    arch_path='x86_64'
fi

if [[ -z "${cusparselt_version}" ]]; then
    echo "Usage: install_cusparselt.sh <cusparselt_version>"
    exit 1
fi

cuda_major_version=${CUDA_VERSION%%.*}
cusparselt_minor=$(echo "${cusparselt_version}" | cut -d. -f2)
# Starting from 0.8.0, NVIDIA ships separate archives per CUDA major version
if [[ "${cusparselt_minor}" -ge 8 ]]; then
    CUSPARSELT_NAME="libcusparse_lt-linux-${arch_path}-${cusparselt_version}_cuda${cuda_major_version}-archive"
else
    CUSPARSELT_NAME="libcusparse_lt-linux-${arch_path}-${cusparselt_version}-archive"
fi

curl --retry 3 -OLs https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-${arch_path}/${CUSPARSELT_NAME}.tar.xz
tar xf ${CUSPARSELT_NAME}.tar.xz
cp -a ${CUSPARSELT_NAME}/include/* /usr/local/cuda/include/
cp -a ${CUSPARSELT_NAME}/lib/* /usr/local/cuda/lib64/
cd ..
rm -rf tmp_cusparselt
ldconfig
