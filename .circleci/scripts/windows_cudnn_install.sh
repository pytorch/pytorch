#!/bin/bash
set -eux -o pipefail

cuda_major_version=${CUDA_VERSION%.*}

if [[ "$cuda_major_version" == "10" ]]; then
    cudnn_installer_name="cudnn-${CUDA_VERSION}-windows10-x64-v7.6.4.38"
elif [[ "$cuda_major_version" == "11" ]]; then
    cudnn_installer_name="cudnn-${CUDA_VERSION}-windows-x64-v8.1.0.77"
else
    echo "CUDNN for CUDA_VERSION $CUDA_VERSION is not supported yet"
    exit 1
fi

cudnn_installer_link="https://ossci-windows.s3.amazonaws.com/${cudnn_installer_name}.zip"

curl --retry 3 -O $cudnn_installer_link
7z x ${cudnn_installer_name}.zip -ocudnn
cp -r cudnn/cuda/* "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/"
rm -rf cudnn
rm -f ${cudnn_installer_name}.zip
