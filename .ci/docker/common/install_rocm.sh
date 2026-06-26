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

    # When ROCM_VERSION=preview, install ROCm from TheRock preview wheels
    if [[ "${ROCM_VERSION}" == "preview" ]]; then
      if [[ -d /opt/rocm ]]; then
        rm -rf /opt/rocm
      fi

      if [[ -z "${THEROCK_NIGHTLY_INDEX_URL:-}" ]]; then
        THEROCK_NIGHTLY_INDEX_URL="https://rocm.nightlies.amd.com/whl-multi-arch/"
      fi
      # FIXME: temporarily pin TheRock nightly ROCm to the June 12 build while
      # the rolling nightly index is unstable.
      THEROCK_NIGHTLY_VERSION="7.14.0a20260612"

      echo "=============================================="
      echo "ROCm Multi-Arch Wheel Installation (TheRock preview)"
      echo "=============================================="
      echo "Index URL: ${THEROCK_NIGHTLY_INDEX_URL}"
      echo "TheRock preview version: ${THEROCK_NIGHTLY_VERSION}"
      echo "=============================================="

      python3 -m pip install \
        --index-url "${THEROCK_NIGHTLY_INDEX_URL}" \
        "rocm[libraries,devel,device-all]==${THEROCK_NIGHTLY_VERSION}"

      # Use the rocm-sdk CLI helper to discover install paths
      ROCM_HOME="$(rocm-sdk path --root)"
      ROCM_BIN="$(rocm-sdk path --bin)"

      echo "ROCM_HOME=${ROCM_HOME}"
      echo "ROCM_BIN=${ROCM_BIN}"

      # theRock bundles system dependencies like libdrm, liblzma in rocm_sysdeps
      ROCM_SYSDEPS="${ROCM_HOME}/lib/rocm_sysdeps"
      ROCM_SYSDEPS_INCLUDE="${ROCM_SYSDEPS}/include"
      ROCM_SYSDEPS_LIB="${ROCM_SYSDEPS}/lib"
      ROCM_SYSDEPS_PKGCONFIG="${ROCM_SYSDEPS_LIB}/pkgconfig"

      # Write environment file (sourced by CI scripts and interactive shells)
      cat > /etc/rocm_env.sh << ROCM_ENV
# ROCm paths discovered from rocm-sdk. Keep this list short: PyTorch's
# LoadHIP.cmake derives CMake package paths, MAGMA_HOME, ROCM_SOURCE_DIR, and
# the TheRock device library path from ROCM_PATH when those env vars are unset.
export ROCM_PATH="${ROCM_HOME}"
export ROCM_HOME="${ROCM_HOME}"
export PATH="${ROCM_BIN}:\${PATH}"
export LD_LIBRARY_PATH="${ROCM_HOME}/lib:\${LD_LIBRARY_PATH:-}"
# theRock system dependencies (libdrm, liblzma, etc.)
export CPLUS_INCLUDE_PATH="${ROCM_SYSDEPS_INCLUDE}:\${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="${ROCM_SYSDEPS_INCLUDE}:\${C_INCLUDE_PATH:-}"
export PKG_CONFIG_PATH="${ROCM_SYSDEPS_PKGCONFIG}:\${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="${ROCM_SYSDEPS_LIB}:\${LD_LIBRARY_PATH}"
export LIBRARY_PATH="${ROCM_SYSDEPS_LIB}:\${LIBRARY_PATH:-}"
# Disable MSLK for theRock preview (not yet supported)
export USE_MSLK=0
ROCM_ENV

      echo "source /etc/rocm_env.sh" >> /etc/bash.bashrc

      echo "=============================================="
      echo "TheRock preview ROCm wheel install complete"
      echo "ROCM_HOME=${ROCM_HOME}"
      echo "=============================================="

      # --- End of theRock preview wheel installation ---
    else
      # =========================================================================
      # Non-preview: install ROCm from repo.radeon.com apt packages
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
        ROCM_VERSION="${ROCM_VERSION}.3"
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

    # Note: rocm-composable-kernel (ck4inductor) is now built as a wheel
    # alongside PyTorch in .ci/pytorch/build.sh and installed at test time

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
