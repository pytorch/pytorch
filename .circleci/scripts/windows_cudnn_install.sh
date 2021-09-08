#!/bin/bash
set -eux -o pipefail

cuda_major_version=${CUDA_VERSION%.*}

case ${CUDA_VERSION} in
    10.1)
        archive_version="v7.6.4.38"
        ;;
    10.2)
        archive_version="v7.6.5.32"
        ;;
    11.1)
        archive_version="v8.0.5.39"
        ;;
    11.3)
        archive_version="v8.2.0.53"
        ;;
    *)
        echo "CUDA_VERSION: ${CUDA_VERSION} not supported yet"
        exit 1
        ;;
esac

cudnn_installer_name="cudnn_installer.zip"
trap EXIT 'rm -rf ${cudnn_installer_name}'
cudnn_installer_link="https://ossci-windows.s3.amazonaws.com/cudnn-${CUDA_VERSION}-windows10-x64-${archive_version}.zip"
cudnn_install_folder="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/"

curl --retry 3 -o "${cudnn_installer_name}" "$cudnn_installer_link"
7z x "${cudnn_installer_name}" -ocudnn
# shellcheck recommends to use '${var:?}/*' to avoid potentially expanding to '/*'
# Remove all of the directories before attempting to copy files
rm -rf "${cudnn_install_folder:?}/*"
cp -rf cudnn/cuda/* "${cudnn_install_folder}"
rm -rf cudnn
rm -f "${cudnn_installer_name}.zip"
