#!/bin/bash

set -ex

ver() {
    printf "%3d%03d%03d%03d" $(echo "$1" | tr '.' ' ');
}

install_ubuntu() {
    apt-get update
    # gpg-agent is not available by default on 20.04
    apt-get install -y --no-install-recommends gpg-agent
    apt-get install -y kmod wget

    # Need the libc++1 and libc++abi1 libraries to allow torch._C to load at runtime
    apt-get install -y libc++1
    apt-get install -y libc++abi1

    # Make sure rocm packages from repo.radeon.com have highest priority
    cat << EOF > /etc/apt/preferences.d/rocm-pin-600
Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 600
EOF

    # Add amdgpu repository
    UBUNTU_VERSION_NAME=`cat /etc/os-release | grep UBUNTU_CODENAME | awk -F= '{print $2}'`
    echo "deb [arch=amd64] https://repo.radeon.com/amdgpu/${ROCM_VERSION}/ubuntu ${UBUNTU_VERSION_NAME} main" > /etc/apt/sources.list.d/amdgpu.list

    # Add rocm repository
    wget -qO - http://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
    local rocm_baseurl="http://repo.radeon.com/rocm/apt/${ROCM_VERSION}"
    echo "deb [arch=amd64] ${rocm_baseurl} ${UBUNTU_VERSION_NAME} main" > /etc/apt/sources.list.d/rocm.list
    apt-get update --allow-insecure-repositories

    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
                   rocm-dev \
                   rocm-libs \
                   rocm-llvm-dev

    # precompiled miopen kernels; search for all unversioned packages
    # if search fails it will abort this script; use true to avoid case where search fails
    MIOPENHIPGFX=$(apt-cache search --names-only miopen-hip-gfx | awk '{print $1}' | grep -F -v . || true)
    if [[ "x${MIOPENHIPGFX}" = x ]]; then
      echo "miopen-hip-gfx package not available" && exit 1
    else
      DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated ${MIOPENHIPGFX}
    fi

    # ROCm 6.0 had a regression where journal_mode was enabled on the kdb files resulting in permission errors at runtime
    for kdb in /opt/rocm/share/miopen/db/*.kdb
    do
        sqlite3 $kdb "PRAGMA journal_mode=off; PRAGMA VACUUM;"
    done

    # ROCm 6.3 had a regression where initializing static code objects had significant overhead
    # ROCm 6.4 did not yet fix the regression, also HIP branch names are different
    if [[ $(ver $ROCM_VERSION) -eq $(ver 6.3) ]] || [[ $(ver $ROCM_VERSION) -eq $(ver 6.4) ]]; then
        if [[ $(ver $ROCM_VERSION) -eq $(ver 6.3) ]]; then
            HIP_BRANCH=rocm-6.3.x
            VER_STR=6.3
        elif [[ $(ver $ROCM_VERSION) -eq $(ver 6.4) ]]; then
            HIP_BRANCH=release/rocm-rel-6.4
            VER_STR=6.4
        fi
        # clr build needs CppHeaderParser but can only find it using conda's python
        /opt/conda/bin/python -m pip install CppHeaderParser
        git clone https://github.com/ROCm/HIP -b $HIP_BRANCH
        HIP_COMMON_DIR=$(readlink -f HIP)
        git clone https://github.com/jeffdaily/clr -b release/rocm-rel-${VER_STR}-statco-hotfix
        mkdir -p clr/build
        pushd clr/build
        cmake .. -DCLR_BUILD_HIP=ON -DHIP_COMMON_DIR=$HIP_COMMON_DIR
        make -j
        cp hipamd/lib/libamdhip64.so.${VER_STR}.* /opt/rocm/lib/libamdhip64.so.${VER_STR}.*
        popd
        rm -rf HIP clr
    fi

    # Cleanup
    apt-get autoclean && apt-get clean
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

install_centos() {

  dnf update -y
  dnf install -y kmod wget openblas-serial dkms kernel-headers kernel-devel epel-release

  # Add amdgpu repository
  local amdgpu_baseurl="https://repo.radeon.com/amdgpu/${ROCM_VERSION}/rhel/9.1/main/x86_64"
  echo "[AMDGPU]" > /etc/yum.repos.d/amdgpu.repo
  echo "name=AMDGPU" >> /etc/yum.repos.d/amdgpu.repo
  echo "baseurl=${amdgpu_baseurl}" >> /etc/yum.repos.d/amdgpu.repo
  echo "enabled=1" >> /etc/yum.repos.d/amdgpu.repo
  echo "gpgcheck=1" >> /etc/yum.repos.d/amdgpu.repo
  echo "gpgkey=http://repo.radeon.com/rocm/rocm.gpg.key" >> /etc/yum.repos.d/amdgpu.repo

  local rocm_baseurl="https://repo.radeon.com/rocm/rhel9/${ROCM_VERSION}/main/"
  echo "[ROCm]" > /etc/yum.repos.d/rocm.repo
  echo "name=ROCm" >> /etc/yum.repos.d/rocm.repo
  echo "baseurl=${rocm_baseurl}" >> /etc/yum.repos.d/rocm.repo
  echo "enabled=1" >> /etc/yum.repos.d/rocm.repo
  echo "gpgcheck=1" >> /etc/yum.repos.d/rocm.repo
  echo "gpgkey=http://repo.radeon.com/rocm/rocm.gpg.key" >> /etc/yum.repos.d/rocm.repo

  dnf update -y --nogpgcheck
  dnf --enablerepo=crb install -y perl-File-BaseDir python3-wheel
  dnf install -y --nogpgcheck rocm-ml-sdk rocm-developer-tools

  # precompiled miopen kernels; search for all unversioned packages
  # if search fails it will abort this script; use true to avoid case where search fails
  MIOPENHIPGFX=$(dnf -q search miopen-hip-gfx | grep miopen-hip-gfx | awk '{print $1}'| grep -F kdb. || true)
  if [[ "x${MIOPENHIPGFX}" = x ]]; then
    echo "miopen-hip-gfx package not available" && exit 1
  else
    dnf install -y ${MIOPENHIPGFX}
  fi

  # ROCm 6.0 had a regression where journal_mode was enabled on the kdb files resulting in permission errors at runtime
  for kdb in /opt/rocm/share/miopen/db/*.kdb
  do
      sqlite3 $kdb "PRAGMA journal_mode=off; PRAGMA VACUUM;"
  done

  # Cleanup
  dnf clean all
  rm -rf /var/cache/yum
  rm -rf /var/lib/yum/yumdb
  rm -rf /var/lib/yum/history
}

# Install Python packages depending on the base OS
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
