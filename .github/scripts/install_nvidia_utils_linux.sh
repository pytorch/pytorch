#!/usr/bin/env bash

set -eou pipefail


DISTRIBUTION=$(. /etc/os-release;echo $ID$VERSION_ID)
DRIVER_VERSION="515.57"
DRIVER_FN="NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run"
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

install_nvidia_driver_amzn2() {
    (
        set -x

        HAS_NVIDIA_DRIVER=0
        # Check if NVIDIA driver has already been installed
        if [ -x "$(command -v nvidia-smi)" ]; then
            # The driver exists, check its version next
            INSTALLED_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)

            if [ "$INSTALLED_DRIVER_VERSION" != "$DRIVER_VERSION" ]; then
                # TODO
                # Remove this after torchrec and FBGEMM have both been updated to use
                # PyTorch NVIDIA installation script instead of using the latest driver
                # from RHEL repo
                HAS_NVIDIA_DRIVER=1
                echo "NVIDIA driver ($INSTALLED_DRIVER_VERSION) has been installed, but we expect to have $DRIVER_VERSION instead. Skipping NVIDIA driver installation for now until torchrec and FBGEMM are updated to use PyTorch NVIDIA installation script instead of RHEL repo"
            else
                HAS_NVIDIA_DRIVER=1
                echo "NVIDIA driver ($INSTALLED_DRIVER_VERSION) has already been installed. Skipping NVIDIA driver installation"
            fi
        fi

        if [ "$HAS_NVIDIA_DRIVER" -eq 0 ]; then
            sudo yum groupinstall -y "Development Tools"
            # ensure our kernel install is the same as our underlying kernel,
            # groupinstall "Development Tools" has a habit of mismatching kernel headers
            sudo yum install -y "kernel-devel-uname-r == $(uname -r)"
            sudo modprobe backlight
            sudo curl -fsL -o /tmp/nvidia_driver "https://s3.amazonaws.com/ossci-linux/nvidia_driver/$DRIVER_FN"
            sudo /bin/bash /tmp/nvidia_driver -s --no-drm || (sudo cat /var/log/nvidia-installer.log && false)
            sudo rm -fv /tmp/nvidia_driver
        fi

        nvidia-smi
    )
}

echo "== Installing nvidia driver ${DRIVER_FN} =="
case "${DISTRIBUTION}" in
    amzn*)
        install_nvidia_driver_amzn2
        ;;
    *)
        echo "ERROR: Unknown distribution ${DISTRIBUTION}"
        exit 1
        ;;
esac

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
