#!/usr/bin/env bash

set -eou pipefail


DISTRIBUTION=$(. /etc/os-release;echo $ID$VERSION_ID)
DRIVER_VERSION="515.76"
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

        # Purge any nvidia driver installed from RHEL repo
        sudo yum remove -y nvidia-driver-latest-dkms

        # Try to gather more information about the runner and its existing NVIDIA driver if any
        echo "Before installing NVIDIA driver"
        lspci
        lsmod
        modinfo nvidia || true

        HAS_NVIDIA_DRIVER=0
        # Check if NVIDIA driver has already been installed
        if [ -x "$(command -v nvidia-smi)" ]; then
            set +e
            # The driver exists, check its version next. Also check only the first GPU if there are more than one of them
            # so that the same driver version is not print over multiple lines
            INSTALLED_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader --id=0)
            NVIDIA_SMI_STATUS=$?

            if [ "$NVIDIA_SMI_STATUS" -ne 0 ] && [ "$NVIDIA_SMI_STATUS" -ne 14 ]; then
                echo "Failed to get NVIDIA driver version ($INSTALLED_DRIVER_VERSION). Continuing"
            elif [ "$INSTALLED_DRIVER_VERSION" != "$DRIVER_VERSION" ]; then
                echo "NVIDIA driver ($INSTALLED_DRIVER_VERSION) has been installed, but we expect to have $DRIVER_VERSION instead. Continuing"
            else
                HAS_NVIDIA_DRIVER=1
                echo "NVIDIA driver ($INSTALLED_DRIVER_VERSION) has already been installed. Skipping NVIDIA driver installation"
            fi
            set -e
        fi

        if [ "$HAS_NVIDIA_DRIVER" -eq 0 ]; then
            sudo yum groupinstall -y "Development Tools"
            # ensure our kernel install is the same as our underlying kernel,
            # groupinstall "Development Tools" has a habit of mismatching kernel headers
            sudo yum install -y "kernel-devel-uname-r == $(uname -r)"
            sudo modprobe backlight
            sudo curl -fsL -o /tmp/nvidia_driver "https://s3.amazonaws.com/ossci-linux/nvidia_driver/$DRIVER_FN"

            set +e
            sudo /bin/bash /tmp/nvidia_driver -s --no-drm
            NVIDIA_INSTALLATION_STATUS=$?

            if [ "$NVIDIA_INSTALLATION_STATUS" -ne 0 ]; then
                sudo cat /var/log/nvidia-installer.log

                NVIDIA_DEVICES=$(lspci -D | grep -i NVIDIA | cut -d' ' -f1)
                # The GPU can get stuck in a failure state if somehow the test crashs the GPU microcode. When this
                # happens, we'll try to reset all NVIDIA devices https://github.com/pytorch/pytorch/issues/88388
                for PCI_ID in "$NVIDIA_DEVICES"; do
                    DEVICE_ENABLED=$(cat /sys/bus/pci/devices/$PCI_ID/enable)

                    echo "Reseting $PCI_ID (enabled state: $DEVICE_ENABLED)"
                    # This requires sudo permission of course
                    echo "1" | sudo tee /sys/bus/pci/devices/$PCI_ID/reset
                    sleep 1
                done
            fi

            sudo rm -fv /tmp/nvidia_driver
            set -e
        fi

        sudo modprobe nvidia || true
        echo "After installing NVIDIA driver"
        lspci
        lsmod
        modinfo nvidia || true

        (
            set +e
            nvidia-smi
            NVIDIA_SMI_STATUS=$?

            # Allowable exit statuses for nvidia-smi, see: https://github.com/NVIDIA/gpu-operator/issues/285
            if [ "$NVIDIA_SMI_STATUS" -eq 0 ] || [ "$NVIDIA_SMI_STATUS" -eq 14 ]; then
                echo "INFO: Ignoring allowed status ${NVIDIA_SMI_STATUS}"
            else
                echo "ERROR: nvidia-smi exited with unresolved status ${NVIDIA_SMI_STATUS}"
                exit ${NVIDIA_SMI_STATUS}
            fi
            set -e
        )
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
