#!/bin/bash
set -eux -o pipefail

if [[ "$CUDA_VERSION" == "10" ]]; then
    cuda_complete_version="10.1"
    cudnn_installer_name="cudnn-10.1-windows10-x64-v7.6.4.38"
elif [[ "$CUDA_VERSION" == "11" ]]; then
    cuda_complete_version="11.0"
    cudnn_installer_name="cudnn-11.0-windows-x64-v8.0.2.39"
else
    echo "CUDNN for CUDA_VERSION $CUDA_VERSION is not supported yet"
    exit 1
fi

cudnn_installer_link="https://ossci-windows.s3.amazonaws.com/${cudnn_installer_name}.zip"

curl --retry 3 -O $cudnn_installer_link
7z x ${cudnn_installer_name}.zip -ocudnn
cp -r cudnn/cuda/* "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${cuda_complete_version}/"
rm -rf cudnn
rm -f ${cudnn_installer_name}.zip
