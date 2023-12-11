#!/bin/bash

set -ex
echo "this is a first test"

if [ -n "${USE_CUSPARSELT}" ]; then
    # cuSPARSELt license: https://docs.nvidia.com/cuda/cusparselt/license.html
    mkdir tmp_cusparselt && cd tmp_cusparselt
    # CUSPARSELT_NAME="libcusparse_lt-linux-x86_64-0.5.0.1-archive.tar.xz"
    if [[ ${CUDA_VERSION:0:4} == "12.1" ]]; then
        CUSPARSELT_NAME="libcusparse_lt-linux-x86_64-0.4.0.7-archive"
        curl --retry 3 -OLs https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/${CUSPARSELT_NAME}.tar.xz
    elif [[ ${CUDA_VERSION:0:4} == "11.8" ]]; then
        CUSPARSELT_NAME="libcusparse_lt-linux-x86_64-0.4.0.7-archive"
        curl --retry 3 -OLs https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/${CUSPARSELT_NAME}.tar.xz
    fi

    tar xf ${CUSPARSELT_NAME}.tar.xz
    cp -a ${CUSPARSELT_NAME}/include/* /usr/include/
    cp -a ${CUSPARSELT_NAME}/include/* /usr/local/cuda/include/
    cp -a ${CUSPARSELT_NAME}/include/* /usr/include/x86_64-linux-gnu/

    cp -a ${CUSPARSELT_NAME}/lib/* /usr/local/cuda/lib64/
    cp -a ${CUSPARSELT_NAME}/lib/* /usr/lib/x86_64-linux-gnu/
    cd ..
    rm -rf tmp_cusparselt
    ldconfig
fi

echo "This is a test"
