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

    # When ROCM_VERSION=nightly, install ROCm from TheRock nightly tarballs
    # Mirrors: https://github.com/ROCm/TheRock/blob/main/dockerfiles/install_rocm_tarball.sh
    if [[ "${ROCM_VERSION}" == "nightly" ]]; then
      apt-get install -y --no-install-recommends pkg-config

      if [[ -d /opt/rocm ]]; then
        rm -rf /opt/rocm
      fi

      # Determine GPU family based on target architecture
      AMDGPU_FAMILY="${THEROCK_AMDGPU_FAMILY:-}"
      if [[ -z "${AMDGPU_FAMILY}" ]]; then
        if [[ "${BUILD_ENVIRONMENT}" == *"gfx950"* ]] || [[ "${PYTORCH_ROCM_ARCH}" == *"gfx950"* ]]; then
          AMDGPU_FAMILY="gfx950-dcgpu"
        else
          AMDGPU_FAMILY="gfx94X-dcgpu"
        fi
      fi

      # Auto-detect latest nightly version if not pinned
      VERSION="${THEROCK_VERSION:-}"
      if [[ -z "${VERSION}" ]]; then
        VERSION=$(curl -fsSL "https://rocm.nightlies.amd.com/tarball/" \
          | grep -oP "therock-dist-linux-${AMDGPU_FAMILY}-\K[^\"]+(?=\.tar\.gz)" \
          | grep -v ADHOCBUILD \
          | sort -V \
          | tail -1)
        if [[ -z "${VERSION}" ]]; then
          echo "Error: Could not find a nightly tarball for ${AMDGPU_FAMILY}"
          exit 1
        fi
      fi

      # URL-encode '+' as '%2B' in VERSION (required for devreleases)
      VERSION_ENCODED="${VERSION//+/%2B}"

      TARBALL_URL="https://rocm.nightlies.amd.com/tarball/therock-dist-linux-${AMDGPU_FAMILY}-${VERSION_ENCODED}.tar.gz"

      echo "=============================================="
      echo "ROCm Tarball Installation"
      echo "=============================================="
      echo "Version:         ${VERSION}"
      echo "AMDGPU Family:   ${AMDGPU_FAMILY}"
      echo "Tarball URL:     ${TARBALL_URL}"
      echo "=============================================="

      # Download tarball
      TARBALL_FILE="/tmp/rocm-tarball.tar.gz"

      echo "Downloading tarball..."
      curl -fsSL -o "$TARBALL_FILE" "$TARBALL_URL" || {
        echo "Error: Failed to download tarball from $TARBALL_URL"
        exit 1
      }

      # Verify download
      if [ ! -f "$TARBALL_FILE" ] || [ ! -s "$TARBALL_FILE" ]; then
        echo "Error: Downloaded file is empty or does not exist"
        exit 1
      fi

      # Install directory is fixed to /opt/rocm-{VERSION}
      ROCM_INSTALL_DIR="/opt/rocm-${VERSION}"

      # Extract tarball to versioned directory
      echo "Extracting tarball to ${ROCM_INSTALL_DIR}..."
      mkdir -p "$ROCM_INSTALL_DIR"
      tar -xzf "$TARBALL_FILE" -C "$ROCM_INSTALL_DIR"

      # Clean up downloaded file
      rm -f "$TARBALL_FILE"
      echo "Tarball extracted and cleaned up"

      # Create symlink /opt/rocm -> /opt/rocm-{VERSION} for compatibility
      ln -sfn "$ROCM_INSTALL_DIR" /opt/rocm
      echo "Created symlink: /opt/rocm -> $ROCM_INSTALL_DIR"

      # Verify bin and lib folder exists after extraction
      echo "Verifying installation..."
      for dir in bin clients include lib libexec share; do
        if [ ! -d "$ROCM_INSTALL_DIR/$dir" ]; then
          echo "Error: ROCm $dir directory not found"
          exit 1
        fi
        echo "ROCm $dir found in $ROCM_INSTALL_DIR/$dir"
      done

      echo "=============================================="
      echo "ROCm installed successfully to $ROCM_INSTALL_DIR"
      echo "ROCM_PATH=$ROCM_INSTALL_DIR"
      echo "PATH should include: $ROCM_INSTALL_DIR/bin"
      echo "=============================================="

      # Write environment file (sourced by CI scripts and interactive shells)
      cat > /etc/rocm_env.sh << ROCM_ENV
# ROCm paths
export ROCM_PATH=/opt/rocm
export ROCM_HOME=/opt/rocm
export ROCM_SOURCE_DIR=/opt/rocm
export ROCM_BIN=/opt/rocm/bin
export ROCM_CMAKE=/opt/rocm
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:\${PATH}
export LD_LIBRARY_PATH=/opt/rocm/lib:\${LD_LIBRARY_PATH:-}
# Sysdeps include paths (libdrm headers, etc.)
export CPLUS_INCLUDE_PATH=/opt/rocm/lib/rocm_sysdeps/include:\${CPLUS_INCLUDE_PATH:-}
export C_INCLUDE_PATH=/opt/rocm/lib/rocm_sysdeps/include:\${C_INCLUDE_PATH:-}
# Device library path
export HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode
export MAGMA_HOME=/opt/rocm/magma
# Tarball bundles sysdeps (libdrm, liblzma, etc.); expose their libs and .pc files
if [ -d /opt/rocm/lib/rocm_sysdeps/lib ]; then
  export LD_LIBRARY_PATH=/opt/rocm/lib/rocm_sysdeps/lib:\${LD_LIBRARY_PATH}
  export PKG_CONFIG_PATH=/opt/rocm/lib/rocm_sysdeps/lib/pkgconfig:\${PKG_CONFIG_PATH:-}
fi
# Disable MSLK for theRock nightly (not yet supported)
export USE_MSLK=0
ROCM_ENV

      echo "source /etc/rocm_env.sh" >> /etc/bash.bashrc

      # --- End of theRock nightly tarball installation ---
    else
      # =========================================================================
      # Non-nightly: install ROCm from repo.radeon.com apt packages
      # =========================================================================

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

    # we want the patch version of 7.2 instead
    if [[ $(ver $ROCM_VERSION) -eq $(ver 7.2) ]]; then
        ROCM_VERSION="${ROCM_VERSION}.1"
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

    # Write environment file (sourced by CI scripts and interactive shells)
    cat > /etc/rocm_env.sh << ROCM_ENV
# ROCm paths
export ROCM_PATH=/opt/rocm
export ROCM_HOME=/opt/rocm
export ROCM_SOURCE_DIR=/opt/rocm
export ROCM_BIN=/opt/rocm/bin
export ROCM_CMAKE=/opt/rocm
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:\${PATH}
export LD_LIBRARY_PATH=/opt/rocm/lib:\${LD_LIBRARY_PATH:-}
# Device library path
export HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode
export MAGMA_HOME=/opt/rocm/magma
ROCM_ENV

    echo "source /etc/rocm_env.sh" >> /etc/bash.bashrc

    # Cleanup
    apt-get autoclean && apt-get clean
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
    fi
}

# Install Python packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    install_ubuntu
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac
