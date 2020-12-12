#!/bin/bash
set -eux -o pipefail

if [[ "$CUDA_VERSION" =~ ^10.* ]]; then
    cudnn_installer_name="cudnn-${cuda_version}-windows10-x64-v7.6.4.38"
elif [[ "$CUDA_VERSION" =~ ^11.* ]]; then
    cudnn_installer_name="cudnn-${cuda_version}-windows-x64-v8.0.5.39"
else
    echo "CUDNN for CUDA_VERSION $CUDA_VERSION is not supported yet"
    exit 1
fi

cudnn_installer_link="https://ossci-windows.s3.amazonaws.com/${cudnn_installer_name}.zip"

curl --retry 3 -O $cudnn_installer_link
7z x ${cudnn_installer_name}.zip -ocudnn
cp -r cudnn/cuda/* "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${cuda_version}/"
rm -rf cudnn
rm -f ${cudnn_installer_name}.zip
