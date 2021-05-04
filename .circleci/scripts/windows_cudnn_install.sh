#!/bin/bash
set -eux -o pipefail

# cuda_installer_name
declare -a installers=(
    "10.1 cudnn-10.1-windows10-x64-v7.6.4.38"
    "11.1 cudnn-11.1-windows-x64-v8.0.5.39"
    "11.2 cudnn-11.2-windows-x64-v8.1.0.77"
    "11.3 cudnn-11.3-windows-x64-v8.2.0.53"
)

for elem in "${installers[@]}"; do
    read -a pair <<< "$elem"  # uses default whitespace IFS
    if [[ "$CUDA_VERSION" == "${pair[0]}" ]]; then
        cudnn_installer_name=${pair[1]}
        break
    fi
done

if [ -z $cudnn_installer_name ]; then
    echo "CUDNN for CUDA_VERSION $CUDA_VERSION is not supported yet"
    exit 1
fi

cudnn_installer_link="https://ossci-windows.s3.amazonaws.com/${cudnn_installer_name}.zip"

curl --retry 3 -O $cudnn_installer_link
7z x ${cudnn_installer_name}.zip -ocudnn
cp -r cudnn/cuda/* "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/"
rm -rf cudnn
rm -f ${cudnn_installer_name}.zip
