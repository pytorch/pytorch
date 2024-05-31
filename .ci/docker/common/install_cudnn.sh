#!/bin/bash

if [[ ${CUDNN_VERSION} == 8 ]]; then
    # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
    mkdir tmp_cudnn
    pushd tmp_cudnn
    if [[ ${CUDA_VERSION:0:4} == "12.4" ]]; then
        CUDNN_NAME="cudnn-linux-x86_64-8.9.7.29_cuda12-archive"
        curl --retry 3 -OLs https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/${CUDNN_NAME}.tar.xz
    elif [[ ${CUDA_VERSION:0:4} == "12.1" ]]; then
        CUDNN_NAME="cudnn-linux-x86_64-8.9.2.26_cuda12-archive"
        curl --retry 3 -OLs https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/${CUDNN_NAME}.tar.xz
    elif [[ ${CUDA_VERSION:0:4} == "11.8" ]]; then
        CUDNN_NAME="cudnn-linux-x86_64-8.7.0.84_cuda11-archive"
        curl --retry 3 -OLs https://developer.download.nvidia.com/compute/redist/cudnn/v8.7.0/local_installers/11.8/${CUDNN_NAME}.tar.xz
    else
        print "Unsupported CUDA version ${CUDA_VERSION}"
        exit 1
    fi

    tar xf ${CUDNN_NAME}.tar.xz
    cp -a ${CUDNN_NAME}/include/* /usr/local/cuda/include/
    cp -a ${CUDNN_NAME}/lib/* /usr/local/cuda/lib64/
    popd
    rm -rf tmp_cudnn
    ldconfig
fi
