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
    echo "install_rocSHMEM.sh: using ROCM install prefix ${rocm_dir}"
    if [[ -f "${rocm_dir}/lib/librocshmem.a" ]]; then
        echo "install_rocSHMEM.sh: librocshmem.a already present in ${rocm_dir}/lib, skipping build"
        return
    fi
    (
        set -x
        curr_dir=$(pwd)
        tmp_dir=$(mktemp -d)

        git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git ${tmp_dir}/rocm-systems
        cd ${tmp_dir}/rocm-systems
        git sparse-checkout set --cone projects/rocshmem
        git checkout ${ROCSHMEM_VERSION}

        cd ${tmp_dir}/rocm-systems/projects/rocshmem
        mkdir build
        cd build
        INSTALL_PREFIX="${rocm_dir}" ../scripts/build_configs/all_backends
        cd ${curr_dir}

    )
}

do_install
