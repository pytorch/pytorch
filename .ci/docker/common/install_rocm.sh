#!/bin/bash

set -ex

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

    # ROCM_VERSION=preview installs ROCm from TheRock wheels via the shared
    # distro-agnostic helper below. The Ubuntu-specific path remains for apt packages.
    if [[ "${ROCM_VERSION}" == "preview" ]]; then
        install_rocm
        return
    fi

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
}

install_rocm() {
    # ROCm from the multi-arch TheRock wheel index. This is distro-agnostic and
    # used by both the AlmaLinux manywheel builder and Ubuntu ROCm preview image.
    # The ROCm SDK unpacks under <site-packages>/_rocm_sdk_*; discover the real
    # install root via `rocm-sdk path` and export it through /etc/rocm_env.sh.
    local therock_index_url rocm_pip_spec
    if [[ "${ROCM_VERSION}" == "preview" ]]; then
        therock_index_url="${THEROCK_NIGHTLY_INDEX_URL:-https://rocm.nightlies.amd.com/whl-multi-arch/}"
        # FIXME: temporarily pin TheRock nightly ROCm to the July 12 build while
        # the rolling nightly index is unstable.
        local therock_nightly_version="${THEROCK_NIGHTLY_VERSION:-7.15.0a20260712}"
        rocm_pip_spec="rocm[libraries,devel,device-all]==${therock_nightly_version}"
    else
        : "${THEROCK_INDEX_URL:?THEROCK_INDEX_URL must be set}"
        : "${ROCM_VERSION:?ROCM_VERSION must be set}"
        therock_index_url="${THEROCK_INDEX_URL}"
        # ROCM_VERSION is the ROCm minor line (e.g. "7.14"); resolve to newest 7.14.x.
        rocm_pip_spec="rocm[libraries,devel,device-all]==${ROCM_VERSION}.*"
    fi

    echo "=============================================="
    echo "ROCm Multi-Arch Wheel Installation (TheRock)"
    echo "Index URL:  ${therock_index_url}"
    echo "ROCm spec:  ${rocm_pip_spec}"
    echo "=============================================="

    # device-all pulls kernels for every supported gfx target (multi-arch wheel);
    # libraries+devel provide the runtime libs + headers/hipcc to compile against ROCm.
    python3 -m pip install --index-url "${therock_index_url}" "${rocm_pip_spec}"

    # Discover the real install root/bin via the rocm-sdk CLI (rocm-sdk-core wheel).
    local rocm_home rocm_bin rocm_sysdeps rocm_sysdeps_include rocm_sysdeps_lib rocm_sysdeps_pkgconfig
    rocm_home="$(rocm-sdk path --root)"
    rocm_bin="$(rocm-sdk path --bin)"
    rocm_sysdeps="${rocm_home}/lib/rocm_sysdeps"
    rocm_sysdeps_include="${rocm_sysdeps}/include"
    rocm_sysdeps_lib="${rocm_sysdeps}/lib"
    rocm_sysdeps_pkgconfig="${rocm_sysdeps_lib}/pkgconfig"

    # Build-time environment file (sourced by CI scripts). Keep the common path
    # exports short; add the preview/sysdeps exports only when that payload exists.
    {
        echo '# ROCm paths discovered from rocm-sdk.'
        printf 'export ROCM_PATH=%q\n' "${rocm_home}"
        printf 'export ROCM_HOME=%q\n' "${rocm_home}"
        printf 'export PATH=%q:${PATH}\n' "${rocm_bin}"
        if [[ "${ROCM_VERSION}" == "preview" ]]; then
            printf 'export LD_LIBRARY_PATH=%q:${LD_LIBRARY_PATH:-}\n' "${rocm_home}/lib"
            echo '# Disable MSLK for theRock preview (not yet supported)'
            echo 'export USE_MSLK=0'
        fi
        if [[ -d "${rocm_sysdeps}" ]]; then
            echo '# TheRock system dependencies (libdrm, liblzma, etc.)'
            printf 'export CPLUS_INCLUDE_PATH=%q:${CPLUS_INCLUDE_PATH:-}\n' "${rocm_sysdeps_include}"
            printf 'export C_INCLUDE_PATH=%q:${C_INCLUDE_PATH:-}\n' "${rocm_sysdeps_include}"
            printf 'export PKG_CONFIG_PATH=%q:${PKG_CONFIG_PATH:-}\n' "${rocm_sysdeps_pkgconfig}"
            printf 'export LD_LIBRARY_PATH=%q:${LD_LIBRARY_PATH:-}\n' "${rocm_sysdeps_lib}"
            printf 'export LIBRARY_PATH=%q:${LIBRARY_PATH:-}\n' "${rocm_sysdeps_lib}"
        fi
    } > /etc/rocm_env.sh

    if [[ -e /etc/bash.bashrc ]]; then
        echo "source /etc/rocm_env.sh" >> /etc/bash.bashrc
    fi
    if [[ -e /etc/bashrc ]]; then
        echo "source /etc/rocm_env.sh" >> /etc/bashrc
    fi

    echo "TheRock ROCm wheel install complete: ROCM_HOME=${rocm_home}"
}

# Install Python packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    install_ubuntu
    ;;
  almalinux)
    install_rocm
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac
