#!/bin/bash
set -xe


# Intel® software for general purpose GPU capabilities.
# Refer to https://dgpu-docs.intel.com/releases/stable_647_21_20230714.html

# Intel® oneAPI Base Toolkit (version 2024.0.0) has been updated to include functional and security updates.
# Refer to https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

# Users should update to the latest version as it becomes available

function install_ubuntu() {
    apt-get update -y
    apt-get install -y gpg-agent wget
    apt-get -y --fix-broken install

    DRIVER_URL=http://mlpc.intel.com/downloads/gpu-new/validation/IPEX/driver/hotfix_agama-ci-devel-775.20/
    DRIVER_ROOT=/opt/
    cd ${DRIVER_ROOT} && mkdir -p ${DRIVER_ROOT}/driver
    wget -r -np --no-proxy ${DRIVER_URL} -P ${DRIVER_ROOT}/driver
    DRIVER_DIR=${DRIVER_ROOT}/driver/mlpc.intel.com/downloads/gpu-new/validation/IPEX/driver/hotfix_agama-ci-devel-775.20/
    # install driver
    if [ ! -z ${DRIVER_DIR} ]; then
        if [ -d ${DRIVER_DIR} ]; then
            while read -r line; do
                file=${DRIVER_DIR}/${line}
                if [ -f $file ]; then
                    if [[ ${file} == *"dkms"* ]] || [[ ${file} == *"/intel-fw-gpu"* ]] || [[ ${file} != *".deb" ]]; then
                echo "skip the ${file}"
            else
                echo "install ${file}"
                        dpkg -i  --force-all ${file}
                    fi
                fi
            done < <(ls -1 ${DRIVER_DIR})
        fi
    fi
    rm -rf ${DRIVER_ROOT}/driver

    # install basekit
    BASEKIT_URL=http://mlpc.intel.com/downloads/gpu-new/validation/IPEX/basekit/l_BaseKit_p_2024.1.0.226_offline.sh
    BASEKIT_ROOT=/opt
    wget ${BASEKIT_URL} -P ${BASEKIT_ROOT} && \
    chmod +x ${BASEKIT_ROOT}/l_BaseKit_*.sh && \
    cd ${BASEKIT_ROOT} && mkdir -p ${BASEKIT_ROOT}/intel/oneapi && \
    sh ./l_BaseKit_*.sh -a --cli --silent --eula accept --install-dir ${BASEKIT_ROOT}/intel/oneapi && \
    rm -rf ${BASEKIT_ROOT}/l_BaseKit_*.sh

    # Cleanup
    apt-get autoclean && apt-get clean
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

function install_centos() {
    dnf install -y 'dnf-command(config-manager)'
    dnf config-manager --add-repo \
        https://repositories.intel.com/gpu/rhel/8.6/production/2328/unified/intel-gpu-8.6.repo
    # To add the EPEL repository needed for DKMS
    dnf -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
        # https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm

    # Create the YUM repository file in the /temp directory as a normal user
    tee > /tmp/oneAPI.repo << EOF
[oneAPI]
name=Intel® oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
EOF

    # Move the newly created oneAPI.repo file to the YUM configuration directory /etc/yum.repos.d
    mv /tmp/oneAPI.repo /etc/yum.repos.d

    # The xpu-smi packages
    dnf install -y flex bison xpu-smi
    # Compute and Media Runtimes
    dnf install -y \
        intel-opencl intel-media intel-mediasdk libmfxgen1 libvpl2\
        level-zero intel-level-zero-gpu mesa-dri-drivers mesa-vulkan-drivers \
        mesa-vdpau-drivers libdrm mesa-libEGL mesa-libgbm mesa-libGL \
        mesa-libxatracker libvpl-tools intel-metrics-discovery \
        intel-metrics-library intel-igc-core intel-igc-cm \
        libva libva-utils intel-gmmlib libmetee intel-gsc intel-ocloc hwinfo clinfo
    # Development packages
    dnf install -y --refresh \
        intel-igc-opencl-devel level-zero-devel intel-gsc-devel libmetee-devel \
        level-zero-devel
    # Install Intel® oneAPI Base Toolkit
    dnf install intel-basekit -y

    # Cleanup
    dnf clean all
    rm -rf /var/cache/yum
    rm -rf /var/lib/yum/yumdb
    rm -rf /var/lib/yum/history
}


# The installation depends on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
    ubuntu)
        install_ubuntu
    ;;
    centos)
        install_centos
    ;;
    *)
        echo "Unable to determine OS..."
        exit 1
    ;;
esac
