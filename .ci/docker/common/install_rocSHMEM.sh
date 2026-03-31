#!/usr/bin/env bash
# Script used only in CD pipeline to build and install rocSHMEM

set -eou pipefail

function do_install() {
    ROCSHMEM_VERSION=ea5c137103f18a9aadd570d09d72e78ec52f0a3a
    rocm_dir="${ROCM_HOME:-}"
    if [[ -z "${rocm_dir}" && -f /etc/rocm_env.sh ]]; then
        source /etc/rocm_env.sh
        rocm_dir="${ROCM_HOME:-}"
    fi
    rocm_dir="${rocm_dir:-/opt/rocm}"
    amdgpu_targets="${ROCSHMEM_AMDGPU_TARGETS:-${PYTORCH_ROCM_ARCH:-}}"

    echo "install_rocSHMEM.sh: building rocm-systems@${ROCSHMEM_VERSION}"
    echo "install_rocSHMEM.sh: using ROCM install prefix ${rocm_dir}"
    if [[ -n "${amdgpu_targets}" ]]; then
        echo "install_rocSHMEM.sh: using explicit GPU targets ${amdgpu_targets}"
    else
        echo "install_rocSHMEM.sh: no explicit GPU targets provided, using rocSHMEM defaults"
    fi
    if [[ -f "${rocm_dir}/lib/librocshmem.a" ]]; then
        echo "install_rocSHMEM.sh: librocshmem.a already present in ${rocm_dir}/lib, rebuilding from pinned source"
    fi
    (
        set -x
        curr_dir=$(pwd)
        tmp_dir=$(mktemp -d)

        sudo apt update -y && sudo apt install -y libibverbs-dev

        git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git "${tmp_dir}/rocm-systems"
        cd "${tmp_dir}/rocm-systems"
        git sparse-checkout set --cone projects/rocshmem
        git checkout "${ROCSHMEM_VERSION}"

        cd "${tmp_dir}/rocm-systems/projects/rocshmem"
        mkdir build
        cd build
        if [[ -n "${amdgpu_targets}" ]]; then
            INSTALL_PREFIX="${rocm_dir}" AMDGPU_TARGETS="${amdgpu_targets}" GPU_TARGETS="${amdgpu_targets}" ../scripts/build_configs/all_backends
        else
            INSTALL_PREFIX="${rocm_dir}" ../scripts/build_configs/all_backends
        fi

        echo "install_rocSHMEM.sh: installed rocSHMEM artifacts"
        ls -l "${rocm_dir}/lib"/librocshmem* || true

        cd "${curr_dir}"
        rm -rf "${tmp_dir}"

    )
}

do_install
