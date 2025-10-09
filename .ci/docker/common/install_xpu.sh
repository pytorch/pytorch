#!/bin/bash
set -xe
# Script used in CI and CD pipeline

# Intel® software for general purpose GPU capabilities.
# Refer to https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html

# Users should update to the latest version as it becomes available

function install_ubuntu() {
    . /etc/os-release
    if [[ ! " jammy noble " =~ " ${VERSION_CODENAME} " ]]; then
        echo "Ubuntu version ${VERSION_CODENAME} not supported"
        exit
    fi

    apt-get update -y
    apt-get install -y gpg-agent wget
    # To add the online network package repository for the GPU Driver
    wget -qO - https://repositories.intel.com/gpu/intel-graphics.key \
        | gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
        https://repositories.intel.com/gpu/ubuntu ${VERSION_CODENAME}${XPU_DRIVER_VERSION} unified" \
        | tee /etc/apt/sources.list.d/intel-gpu-${VERSION_CODENAME}.list
    # To add the online network network package repository for the Intel Support Packages
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor > /usr/share/keyrings/oneapi-archive-keyring.gpg.gpg
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg.gpg] \
        https://apt.repos.intel.com/oneapi all main" \
        | tee /etc/apt/sources.list.d/oneAPI.list

    # Update the packages list and repository index
    apt-get update

    # The xpu-smi packages
    apt-get install -y flex bison xpu-smi

    # Compute and Media Runtimes
    if [[ " ${VERSION_CODENAME} " =~ " noble " ]]; then
        apt-get install -y \
            intel-opencl-icd libze-intel-gpu1 libze1 \
            intel-media-va-driver-non-free libmfx-gen1 libvpl2 \
            libegl-mesa0 libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
            libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
            mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo hwinfo clinfo intel-ocloc
    else # jammy
        apt-get install -y \
            intel-opencl-icd libze-intel-gpu1 libze1 \
            intel-media-va-driver-non-free libmfx-gen1 libvpl2 \
            libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
            libglapi-mesa libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
            mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo hwinfo clinfo intel-ocloc
    fi
    # Development Packages
    apt-get install -y libigc-dev intel-igc-cm libigdfcl-dev libigfxcmrt-dev libze-dev

    # Install Intel Support Packages
    apt-get install -y ${XPU_PACKAGES}

    # Cleanup
    apt-get autoclean && apt-get clean
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

function install_rhel() {
    . /etc/os-release
    if [[ "${ID}" == "rhel" ]]; then
        if [[ ! " 8.8 8.10 9.0 9.2 9.3 " =~ " ${VERSION_ID} " ]]; then
            echo "RHEL version ${VERSION_ID} not supported"
            exit
        fi
    elif [[ "${ID}" == "almalinux" ]]; then
        # Workaround for almalinux8 which used by quay.io/pypa/manylinux_2_28_x86_64
        VERSION_ID="8.8"
    fi

    dnf install -y 'dnf-command(config-manager)'
    # To add the online network package repository for the GPU Driver
    dnf config-manager --add-repo \
        https://repositories.intel.com/gpu/rhel/${VERSION_ID}${XPU_DRIVER_VERSION}/unified/intel-gpu-${VERSION_ID}.repo
    # To add the online network network package repository for the Intel Support Packages
    tee > /etc/yum.repos.d/oneAPI.repo << EOF
[oneAPI]
name=Intel for Pytorch GPU dev repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
EOF

    # Install Intel Support Packages
    yum install -y ${XPU_PACKAGES}
    # The xpu-smi packages
    dnf install -y xpu-smi
    # Compute and Media Runtimes
    dnf install --skip-broken -y \
        intel-opencl intel-media intel-mediasdk libmfxgen1 libvpl2\
        level-zero intel-level-zero-gpu mesa-dri-drivers mesa-vulkan-drivers \
        mesa-vdpau-drivers libdrm mesa-libEGL mesa-libgbm mesa-libGL \
        mesa-libxatracker libvpl-tools intel-metrics-discovery \
        intel-metrics-library intel-igc-core intel-igc-cm \
        libva libva-utils intel-gmmlib libmetee intel-gsc intel-ocloc
    # Development packages
    dnf install -y --refresh \
        intel-igc-opencl-devel level-zero-devel intel-gsc-devel libmetee-devel \
        level-zero-devel

    # Cleanup
    dnf clean all
    rm -rf /var/cache/yum
    rm -rf /var/lib/yum/yumdb
    rm -rf /var/lib/yum/history
}

function install_sles() {
    . /etc/os-release
    VERSION_SP=${VERSION_ID//./sp}
    if [[ ! " 15sp4 15sp5 " =~ " ${VERSION_SP} " ]]; then
        echo "SLES version ${VERSION_ID} not supported"
        exit
    fi

    # To add the online network package repository for the GPU Driver
    zypper addrepo -f -r \
        https://repositories.intel.com/gpu/sles/${VERSION_SP}${XPU_DRIVER_VERSION}/unified/intel-gpu-${VERSION_SP}.repo
    rpm --import https://repositories.intel.com/gpu/intel-graphics.key
    # To add the online network network package repository for the Intel Support Packages
    zypper addrepo https://yum.repos.intel.com/oneapi oneAPI
    rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

    # The xpu-smi packages
    zypper install -y lsb-release flex bison xpu-smi
    # Compute and Media Runtimes
    zypper install -y intel-level-zero-gpu level-zero intel-gsc intel-opencl intel-ocloc \
        intel-media-driver libigfxcmrt7 libvpl2 libvpl-tools libmfxgen1 libmfx1
    # Development packages
    zypper install -y libigdfcl-devel intel-igc-cm libigfxcmrt-devel level-zero-devel

    # Install Intel Support Packages
    zypper install -y ${XPU_PACKAGES}

}

# Default use GPU driver rolling releases
XPU_DRIVER_VERSION=""
if [[ "${XPU_DRIVER_TYPE,,}" == "lts" ]]; then
    # Use GPU driver LTS releases
    XPU_DRIVER_VERSION="/lts/2523"
fi

# Default use Intel® oneAPI Deep Learning Essentials 2025.1
if [[ "$XPU_VERSION" == "2025.2" ]]; then
    XPU_PACKAGES="intel-deep-learning-essentials-2025.2"
else
    XPU_PACKAGES="intel-deep-learning-essentials-2025.1"
fi

# The installation depends on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
    ubuntu)
        install_ubuntu
    ;;
    rhel|almalinux)
        install_rhel
    ;;
    sles)
        install_sles
    ;;
    *)
        echo "Unable to determine OS..."
        exit 1
    ;;
esac
