#!/bin/bash

if [[ -n "${CUDNN_VERSION}" ]]; then
    # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
    mkdir tmp_cudnn
    pushd tmp_cudnn
    if [[ ${CUDA_VERSION:0:2} == "12" ]]; then
        CUDNN_NAME="cudnn-linux-x86_64-9.1.0.70_cuda12-archive"
    elif [[ ${CUDA_VERSION:0:2} == "11" ]]; then
        CUDNN_NAME="cudnn-linux-x86_64-9.1.0.70_cuda11-archive"
    else
        print "Unsupported CUDA version ${CUDA_VERSION}"
        exit 1
    fi
    curl --retry 3 -OLs https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/${CUDNN_NAME}.tar.xz
    tar xf ${CUDNN_NAME}.tar.xz
    cp -a ${CUDNN_NAME}/include/* /usr/local/cuda/include/
    cp -a ${CUDNN_NAME}/lib/* /usr/local/cuda/lib64/
    popd
    rm -rf tmp_cudnn
    ldconfig
fi
