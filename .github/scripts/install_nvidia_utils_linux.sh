#!/usr/bin/env bash

set -eou pipefail

DISTRIBUTION=$(. /etc/os-release;echo $ID$VERSION_ID) \
DRIVER_FN="NVIDIA-Linux-x86_64-460.39.run"
YUM_REPO_URL="https://nvidia.github.io/nvidia-docker/${DISTRIBUTION}/nvidia-docker.repo"

install_nvidia_docker2_amzn2() {
    (
        set -x
        # Needed for yum-config-manager
        sudo yum install -y yum-utils
        sudo yum-config-manager --add-repo "${YUM_REPO_URL}"
        sudo yum install -y nvidia-docker2
        sudo systemctl restart docker
    )
}

install_nvidia_driver() {
    (
        set -x
        sudo yum groupinstall -y "Development Tools"
        curl -fsL -o nvidia_driver "https://s3.amazonaws.com/ossci-linux/nvidia_driver/$DRIVER_FN"
        sudo /bin/bash nvidia_driver -s --no-drm || (sudo cat /var/log/nvidia-installer.log && false)
        nvidia-smi
    )
}

# Install container toolkit based on distribution
echo "== Installing nvidia container toolkit for ${DISTRIBUTION} =="
case "${DISTRIBUTION}" in
    amzn*)
        install_nvidia_docker2_amzn2
        ;;
    *)
        echo "ERROR: Unknown distribution ${DISTRIBUTION}"
        exit 1
        ;;
esac

echo "== Installing nvidia driver ${DRIVER_FN} =="
install_nvidia_driver
