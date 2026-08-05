#!/bin/bash

set -ex

install_ubuntu() {
    apt-get update
    # kmod is used by GPU diagnostics; libc++ lets torch._C load at runtime.
    apt-get install -y --no-install-recommends kmod libc++1 libc++abi1

    install_rocm

    apt-get autoclean && apt-get clean
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

install_rocm() {
    # ROCm from the multi-arch TheRock wheel index for Ubuntu ROCm CI images.
    # The ROCm SDK unpacks under <site-packages>/_rocm_sdk_*; discover the real
    # install root via `rocm-sdk path` and export it through /etc/rocm_env.sh.
    : "${THEROCK_INDEX_URL:?THEROCK_INDEX_URL must be set}"
    : "${ROCM_VERSION:?ROCM_VERSION must be set}"

    # Release lines track the latest patch, while preview builds stay pinned to
    # their fully specified alpha version.
    local rocm_pip_version="${ROCM_VERSION}"
    if [[ "${ROCM_VERSION}" =~ ^[0-9]+\.[0-9]+$ ]]; then
        rocm_pip_version="${ROCM_VERSION}.*"
    fi

    echo "=============================================="
    echo "ROCm Multi-Arch Wheel Installation (TheRock)"
    echo "Index URL:  ${THEROCK_INDEX_URL}"
    echo "ROCm version: ${ROCM_VERSION}"
    echo "ROCm spec:  rocm[libraries,devel,device-all]==${rocm_pip_version}"
    echo "=============================================="

    # device-all pulls kernels for every supported gfx target (multi-arch wheel);
    # libraries+devel provide the runtime libs + headers/hipcc to compile against ROCm.
    python3 -m pip install --index-url "${THEROCK_INDEX_URL}" "rocm[libraries,devel,device-all]==${rocm_pip_version}"

    # Discover the real install root/bin via the rocm-sdk CLI (rocm-sdk-core wheel).
    local rocm_home rocm_bin
    rocm_home="$(rocm-sdk path --root)"
    rocm_bin="$(rocm-sdk path --bin)"

    # Build-time environment file sourced by CI scripts and interactive shells.
    {
        echo '# ROCm paths discovered from rocm-sdk.'
        printf 'export ROCM_PATH=%q\n' "${rocm_home}"
        printf 'export ROCM_HOME=%q\n' "${rocm_home}"
        printf 'export PATH=%q:${PATH}\n' "${rocm_bin}"
        # Standalone CMake projects such as rocm-origami call find_package(hip)
        # directly instead of using PyTorch's ROCM_PATH-aware LoadHIP.cmake.
        printf 'export CMAKE_PREFIX_PATH=%q:${CMAKE_PREFIX_PATH:-}\n' "${rocm_home}"
        printf 'export LD_LIBRARY_PATH=%q:${LD_LIBRARY_PATH:-}\n' "${rocm_home}/lib"
        if [[ -n "${USE_MSLK:-}" ]]; then
            printf 'export USE_MSLK=%q\n' "${USE_MSLK}"
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

install_almalinux() {
    # ROCm from the multi-arch TheRock wheel index (used by the manywheel builder
    # image). Distro-agnostic pip install; no OS packages. The ROCm SDK unpacks
    # under <site-packages>/_rocm_sdk_* and its real install root is discovered via
    # `rocm-sdk path` and exported through /etc/rocm_env.sh (no /opt/rocm symlink);
    # build_env_setup.py / repair_wheel.py discover ROCM_HOME from there. Mirrors #188429.
    : "${THEROCK_INDEX_URL:?THEROCK_INDEX_URL must be set}"
    : "${ROCM_VERSION:?ROCM_VERSION must be set}"
    # ROCM_VERSION is the minor line (e.g. "7.14"); pip resolves to newest 7.14.x.
    local ROCM_PIP_SPEC="rocm[libraries,devel,device-all]==${ROCM_VERSION}.*"

    echo "=============================================="
    echo "ROCm Multi-Arch Wheel Installation (TheRock)"
    echo "Index URL:  ${THEROCK_INDEX_URL}"
    echo "ROCm spec:  ${ROCM_PIP_SPEC}"
    echo "=============================================="

    # device-all pulls kernels for every supported gfx target (multi-arch wheel);
    # libraries+devel provide the runtime libs + headers/hipcc to compile against ROCm.
    python3 -m pip install --index-url "${THEROCK_INDEX_URL}" "${ROCM_PIP_SPEC}"

    # Discover the real install root/bin via the rocm-sdk CLI (rocm-sdk-core wheel).
    local ROCM_HOME ROCM_BIN
    ROCM_HOME="$(rocm-sdk path --root)"
    ROCM_BIN="$(rocm-sdk path --bin)"

    # Build-time environment file (sourced by CI scripts): ROCM_PATH/PATH let the
    # build find ROCm + hipcc. No LD_LIBRARY_PATH: the produced wheel resolves ROCm
    # at runtime via RPATH (see repair_wheel.py), and TheRock's own binaries carry
    # their RPATHs, so the build does not need it either.
    cat > /etc/rocm_env.sh << ROCM_ENV
export ROCM_PATH="${ROCM_HOME}"
export ROCM_HOME="${ROCM_HOME}"
export PATH="${ROCM_BIN}:\${PATH}"
ROCM_ENV
    echo "source /etc/rocm_env.sh" >> /etc/bashrc || true

    echo "TheRock ROCm wheel install complete: ROCM_HOME=${ROCM_HOME}"
}

# Install Python packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    install_ubuntu
    ;;
  almalinux)
    install_almalinux
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac
