#!/bin/bash

if [[ ${CUDNN_VERSION} == 8 ]]; then
    # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
    mkdir tmp_cudnn && cd tmp_cudnn
    CUDNN_NAME="cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive"
    curl -OLs  https://developer.download.nvidia.com/compute/redist/cudnn/v8.3.2/local_installers/11.5/${CUDNN_NAME}.tar.xz
    tar xf ${CUDNN_NAME}.tar.xz
    cp -a ${CUDNN_NAME}/include/* /usr/include/
    cp -a ${CUDNN_NAME}/include/* /usr/local/cuda/include/
    cp -a ${CUDNN_NAME}/include/* /usr/include/x86_64-linux-gnu/

    cp -a ${CUDNN_NAME}/lib/* /usr/local/cuda/lib64/
    cp -a ${CUDNN_NAME}/lib/* /usr/lib/x86_64-linux-gnu/
    cd ..
    rm -rf tmp_cudnn
    ldconfig
fi
