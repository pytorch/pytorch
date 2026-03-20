#!/usr/bin/env bash
# Script used only in CD pipeline to build and install rocSHMEM

set -eou pipefail

function do_install() {
    ROCSHMEM_VERSION=7a14c59a186016cf8cdebf5afc706f5ede18c922
    rocm_dir="${ROCM_HOME:-}"
    if [[ -z "${rocm_dir}" && -f /etc/rocm_env.sh ]]; then
        source /etc/rocm_env.sh
        rocm_dir="${ROCM_HOME:-}"
    fi
    rocm_dir="${rocm_dir:-/opt/rocm}"
    echo "install_rocSHMEM.sh: using ROCM install prefix ${rocm_dir}"
    (
        set -x
        curr_dir=$(pwd)
        tmp_dir=$(mktemp -d)

        sudo apt update -y && sudo apt install -y libibverbs-dev

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
