#!/bin/bash

set -ex

# for pip_install function
source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

ROCM_COMPOSABLE_KERNEL_VERSION="$(cat $(dirname $0)/../ci_commit_pins/rocm-composable-kernel.txt)"

ver() {
    printf "%3d%03d%03d%03d" $(echo "$1" | tr '.' ' ');
}

install_ubuntu() {
    apt-get update
    # gpg-agent is not available by default
    apt-get install -y --no-install-recommends gpg-agent
    if [[ $(ver $UBUNTU_VERSION) -ge $(ver 22.04) ]]; then
        echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
            | sudo tee /etc/apt/preferences.d/rocm-pin-600
    fi
    apt-get install -y kmod
    apt-get install -y wget

    # Need the libc++1 and libc++abi1 libraries to allow torch._C to load at runtime
    apt-get install -y libc++1
    apt-get install -y libc++abi1

    # When ROCM_VERSION=nightly, install ROCm from TheRock nightly tarball
    if [[ "${ROCM_VERSION}" == "nightly" ]]; then
      echo "install_rocm.sh: installing ROCm from TheRock nightly tarball"

      # Clean any previous ROCm installation in the base CI image.
      if [[ -d /opt/rocm ]] || [[ -L /opt/rocm ]]; then
        echo "Removing existing /opt/rocm from base image"
        rm -rf /opt/rocm*
      fi

      # Determine AMDGPU_FAMILY based on GPU architecture
      if [[ "${BUILD_ENVIRONMENT}" == *"gfx950"* ]] || [[ "${PYTORCH_ROCM_ARCH}" == *"gfx950"* ]]; then
        AMDGPU_FAMILY="gfx950-dcgpu"
        echo "Detected gfx950 architecture - using MI350 theRock nightly tarball"
      else
        AMDGPU_FAMILY="gfx94X-dcgpu"
        echo "Using gfx94X (MI300) theRock nightly tarball"
      fi

      # Get theRock version - use THEROCK_VERSION env var or fetch latest from nightly index
      if [[ -z "${THEROCK_VERSION:-}" ]]; then
        echo "Fetching latest theRock nightly version..."
        # Get the latest tarball version from the nightlies index
        THEROCK_VERSION=$(curl -sL "https://rocm.nightlies.amd.com/tarball/" | \
          grep -oP "therock-dist-linux-${AMDGPU_FAMILY}-\K[0-9]+\.[0-9]+\.[0-9]+a[0-9]+" | \
          sort -V | tail -1)
        if [[ -z "${THEROCK_VERSION}" ]]; then
          echo "Error: Could not determine latest theRock version"
          exit 1
        fi
      fi

      echo "TheRock Version: ${THEROCK_VERSION}"
      echo "AMDGPU Family: ${AMDGPU_FAMILY}"

      TARBALL_URL="https://rocm.nightlies.amd.com/tarball/therock-dist-linux-${AMDGPU_FAMILY}-${THEROCK_VERSION}.tar.gz"
      echo "Tarball URL: ${TARBALL_URL}"

      # Download tarball
      TARBALL_FILE="/tmp/rocm-tarball.tar.gz"
      echo "Downloading theRock tarball..."
      curl -fsSL -o "$TARBALL_FILE" "$TARBALL_URL" || {
        echo "Error: Failed to download tarball from $TARBALL_URL"
        exit 1
      }

      # Verify download
      if [[ ! -f "$TARBALL_FILE" ]] || [[ ! -s "$TARBALL_FILE" ]]; then
        echo "Error: Downloaded file is empty or does not exist"
        exit 1
      fi

      # Install to /opt/rocm-{VERSION}
      ROCM_INSTALL_DIR="/opt/rocm-${THEROCK_VERSION}"
      echo "Extracting tarball to ${ROCM_INSTALL_DIR}..."
      mkdir -p "$ROCM_INSTALL_DIR"
      tar -xzf "$TARBALL_FILE" -C "$ROCM_INSTALL_DIR"
      rm -f "$TARBALL_FILE"

      # Create symlink /opt/rocm -> /opt/rocm-{VERSION}
      ln -sfn "$ROCM_INSTALL_DIR" /opt/rocm
      echo "Created symlink: /opt/rocm -> $ROCM_INSTALL_DIR"

      # Verify installation
      echo "Verifying installation..."
      for dir in bin include lib share; do
        if [[ ! -d "$ROCM_INSTALL_DIR/$dir" ]]; then
          echo "Warning: ROCm $dir directory not found in $ROCM_INSTALL_DIR"
        else
          echo "ROCm $dir found in $ROCM_INSTALL_DIR/$dir"
        fi
      done

      # Write environment to file that can be sourced by CI scripts and users
      cat > /etc/rocm_env.sh << 'ROCM_ENV'
# ROCm paths (theRock tarball installation)
export ROCM_PATH=/opt/rocm
export ROCM_HOME=/opt/rocm
export ROCM_SOURCE_DIR=/opt/rocm
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:${PATH}
export CMAKE_PREFIX_PATH=/opt/rocm:${CMAKE_PREFIX_PATH:-}
# Device library paths
export HIP_DEVICE_LIB_PATH=/opt/rocm/lib/llvm/amdgcn/bitcode
export ROCM_DEVICE_LIB_PATH=/opt/rocm/lib/llvm/amdgcn/bitcode
export MAGMA_HOME=/opt/rocm/magma
# Disable features not yet supported in theRock nightly
export USE_MSLK=0
ROCM_ENV

      # Append to bash.bashrc so interactive shells get the env vars
      echo "source /etc/rocm_env.sh" >> /etc/bash.bashrc

      echo "install_rocm.sh: TheRock nightly ROCm tarball install complete"
      echo "ROCM_PATH=/opt/rocm"
      exit 0
    fi

    # Make sure rocm packages from repo.radeon.com have highest priority
    cat << EOF > /etc/apt/preferences.d/rocm-pin-600
Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 600
EOF

    # we want the patch version of 6.4 instead
    if [[ $(ver $ROCM_VERSION) -eq $(ver 6.4) ]]; then
        ROCM_VERSION="${ROCM_VERSION}.2"
    fi

    # Default url values
    rocm_baseurl="http://repo.radeon.com/rocm/apt/${ROCM_VERSION}"
    UBUNTU_VERSION_NAME=`cat /etc/os-release | grep UBUNTU_CODENAME | awk -F= '{print $2}'`

    # Add rocm repository
    wget -qO - http://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
    echo "deb [arch=amd64] ${rocm_baseurl} ${UBUNTU_VERSION_NAME} main" > /etc/apt/sources.list.d/rocm.list
    apt-get update --allow-insecure-repositories

    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
                   rocm-dev \
                   rocm-utils \
                   rocm-libs \
                   rccl \
                   rocprofiler-dev \
                   roctracer-dev \
                   amd-smi-lib

    if [[ $(ver $ROCM_VERSION) -ge $(ver 6.1) ]]; then
        DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated rocm-llvm-dev
    fi

    if [[ $(ver $ROCM_VERSION) -lt $(ver 7.1) ]]; then
      # precompiled miopen kernels added in ROCm 3.5, renamed in ROCm 5.5, removed in ROCm 7.1
      # search for all unversioned packages
      # if search fails it will abort this script; use true to avoid case where search fails
      MIOPENHIPGFX=$(apt-cache search --names-only miopen-hip-gfx | awk '{print $1}' | grep -F -v . || true)
      if [[ "x${MIOPENHIPGFX}" = x ]]; then
        echo "miopen-hip-gfx package not available" && exit 1
      else
        DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated ${MIOPENHIPGFX}
      fi
    fi

    # ROCm 6.0 had a regression where journal_mode was enabled on the kdb files resulting in permission errors at runtime
    for kdb in /opt/rocm/share/miopen/db/*.kdb
    do
        sqlite3 $kdb "PRAGMA journal_mode=off; PRAGMA VACUUM;"
    done

    # ROCm 6.3 had a regression where initializing static code objects had significant overhead
    # CI no longer builds for ROCm 6.3, but
    # ROCm 6.4 did not yet fix the regression, also HIP branch names are different
    if [[ $(ver $ROCM_VERSION) -ge $(ver 6.4) ]] && [[ $(ver $ROCM_VERSION) -lt $(ver 7.0) ]]; then
        if [[ $(ver $ROCM_VERSION) -eq $(ver 6.4.2) ]]; then
            HIP_TAG=rocm-6.4.2
            CLR_HASH=74d78ba3ac4bac235d02bcb48511c30b5cfdd457  # branch release/rocm-rel-6.4.2-statco-hotfix
        elif [[ $(ver $ROCM_VERSION) -eq $(ver 6.4.1) ]]; then
            HIP_TAG=rocm-6.4.1
            CLR_HASH=efe6c35790b9206923bfeed1209902feff37f386  # branch release/rocm-rel-6.4.1-statco-hotfix
        elif [[ $(ver $ROCM_VERSION) -eq $(ver 6.4) ]]; then
            HIP_TAG=rocm-6.4.0
            CLR_HASH=600f5b0d2baed94d5121e2174a9de0851b040b0c  # branch release/rocm-rel-6.4-statco-hotfix
        fi
        # clr build needs CppHeaderParser but can only find it using conda's python
        python -m pip install CppHeaderParser
        git clone https://github.com/ROCm/HIP -b $HIP_TAG
        HIP_COMMON_DIR=$(readlink -f HIP)
        git clone https://github.com/jeffdaily/clr
        pushd clr
        git checkout $CLR_HASH
        popd
        mkdir -p clr/build
        pushd clr/build
        # Need to point CMake to the correct python installation to find CppHeaderParser
        cmake .. -DPython3_EXECUTABLE=/opt/conda/envs/py_${ANACONDA_PYTHON_VERSION}/bin/python3 -DCLR_BUILD_HIP=ON -DHIP_COMMON_DIR=$HIP_COMMON_DIR
        make -j
        cp hipamd/lib/libamdhip64.so.6.4.* /opt/rocm/lib/libamdhip64.so.6.4.*
        popd
        rm -rf HIP clr
    fi

    pip_install "git+https://github.com/rocm/composable_kernel@$ROCM_COMPOSABLE_KERNEL_VERSION"

    # Cleanup
    apt-get autoclean && apt-get clean
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

install_centos() {

  yum update -y
  yum install -y kmod
  yum install -y wget
  yum install -y openblas-devel

  yum install -y epel-release
  yum install -y dkms kernel-headers-`uname -r` kernel-devel-`uname -r`

  # Add amdgpu repository
  local amdgpu_baseurl
  if [[ $OS_VERSION == 9 ]]; then
      amdgpu_baseurl="https://repo.radeon.com/amdgpu/${ROCM_VERSION}/rhel/9.0/main/x86_64"
  else
      amdgpu_baseurl="https://repo.radeon.com/amdgpu/${ROCM_VERSION}/rhel/7.9/main/x86_64"
  fi
  echo "[AMDGPU]" > /etc/yum.repos.d/amdgpu.repo
  echo "name=AMDGPU" >> /etc/yum.repos.d/amdgpu.repo
  echo "baseurl=${amdgpu_baseurl}" >> /etc/yum.repos.d/amdgpu.repo
  echo "enabled=1" >> /etc/yum.repos.d/amdgpu.repo
  echo "gpgcheck=1" >> /etc/yum.repos.d/amdgpu.repo
  echo "gpgkey=http://repo.radeon.com/rocm/rocm.gpg.key" >> /etc/yum.repos.d/amdgpu.repo

  local rocm_baseurl="http://repo.radeon.com/rocm/yum/${ROCM_VERSION}"
  echo "[ROCm]" > /etc/yum.repos.d/rocm.repo
  echo "name=ROCm" >> /etc/yum.repos.d/rocm.repo
  echo "baseurl=${rocm_baseurl}" >> /etc/yum.repos.d/rocm.repo
  echo "enabled=1" >> /etc/yum.repos.d/rocm.repo
  echo "gpgcheck=1" >> /etc/yum.repos.d/rocm.repo
  echo "gpgkey=http://repo.radeon.com/rocm/rocm.gpg.key" >> /etc/yum.repos.d/rocm.repo

  yum update -y

  yum install -y \
                   rocm-dev \
                   rocm-utils \
                   rocm-libs \
                   rccl \
                   rocprofiler-dev \
                   roctracer-dev \
                   amd-smi-lib

  # precompiled miopen kernels; search for all unversioned packages
  # if search fails it will abort this script; use true to avoid case where search fails
  MIOPENHIPGFX=$(yum -q search miopen-hip-gfx | grep miopen-hip-gfx | awk '{print $1}'| grep -F kdb. || true)
  if [[ "x${MIOPENHIPGFX}" = x ]]; then
    echo "miopen-hip-gfx package not available" && exit 1
  else
    yum install -y ${MIOPENHIPGFX}
  fi

  # ROCm 6.0 had a regression where journal_mode was enabled on the kdb files resulting in permission errors at runtime
  for kdb in /opt/rocm/share/miopen/db/*.kdb
  do
      sqlite3 $kdb "PRAGMA journal_mode=off; PRAGMA VACUUM;"
  done

  pip_install "git+https://github.com/rocm/composable_kernel@$ROCM_COMPOSABLE_KERNEL_VERSION"

  # Cleanup
  yum clean all
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
