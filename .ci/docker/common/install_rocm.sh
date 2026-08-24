#!/bin/bash

set -ex

install_ubuntu() {
    apt-get update
    # kmod is used by GPU diagnostics; libc++ lets torch._C load at runtime.
    apt-get install -y --no-install-recommends kmod libc++1 libc++abi1
    # FIXME: Needed for rocSHMEM in ROCm7.14 since it had a dependency on libnuma.so
    apt-get install -y libnuma-dev

    install_rocm

    # Standalone CMake projects such as rocm-origami call find_package(hip)
    # directly instead of using PyTorch's ROCM_PATH-aware LoadHIP.cmake.
    {
        printf 'export CMAKE_PREFIX_PATH=%q:${CMAKE_PREFIX_PATH:-}\n' "${ROCM_HOME}"
        printf 'export LD_LIBRARY_PATH=%q:${LD_LIBRARY_PATH:-}\n' "${ROCM_HOME}/lib"
        if [[ -n "${USE_MSLK:-}" ]]; then
            printf 'export USE_MSLK=%q\n' "${USE_MSLK}"
        fi
    } >> /etc/rocm_env.sh

    if [[ -e /etc/bash.bashrc ]]; then
        echo "source /etc/rocm_env.sh" >> /etc/bash.bashrc
    fi
    if [[ -e /etc/bashrc ]]; then
        echo "source /etc/rocm_env.sh" >> /etc/bashrc
    fi

    apt-get autoclean && apt-get clean
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

install_rocm() {
    # ROCm from the multi-arch TheRock wheel index.
    # The ROCm SDK unpacks under <site-packages>/_rocm_sdk_*; discover the real
    # install root via `rocm-sdk path` and export it through /etc/rocm_env.sh.
    : "${THEROCK_INDEX_URL:?THEROCK_INDEX_URL must be set}"
    : "${ROCM_VERSION:?ROCM_VERSION must be set}"

    # Major.minor pins (e.g. 7.14) accept later patches via ==7.14.*.
    # Fully specified preview pins (e.g. 7.15.0a20260712) stay exact; PEP 440
    # rejects ==7.15.0a20260712.*.
    rocm_pip_version="${ROCM_VERSION}"
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
    ROCM_HOME="$(rocm-sdk path --root)"
    ROCM_BIN="$(rocm-sdk path --bin)"

    # Common build-time environment sourced by CI scripts.
    {
        echo '# ROCm paths discovered from rocm-sdk.'
        printf 'export ROCM_PATH=%q\n' "${ROCM_HOME}"
        printf 'export ROCM_HOME=%q\n' "${ROCM_HOME}"
        printf 'export PATH=%q:${PATH}\n' "${ROCM_BIN}"
    } > /etc/rocm_env.sh

    echo "TheRock ROCm wheel install complete: ROCM_HOME=${ROCM_HOME}"
}

install_almalinux() {
    # The manywheel build intentionally uses only the common ROCm environment.
    # Its produced wheel resolves ROCm at runtime via RPATH (see repair_wheel.py),
    # so unlike Ubuntu CI it must not add ROCm to LD_LIBRARY_PATH.
    install_rocm
    echo "source /etc/rocm_env.sh" >> /etc/bashrc || true
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
