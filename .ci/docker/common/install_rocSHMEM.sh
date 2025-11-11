#!/usr/bin/env bash
# Script used only in CD pipeline to build and install rocSHMEM

set -eou pipefail

function do_install() {
    ROCSHMEM_VERSION=07517b17db9581296ee6b161612b894ae56b8341
    rocm_dir="${ROCM_HOME:-/opt/rocm}"
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
        INSTALL_PREFIX=${rocm_dir} ../scripts/build_configs/all_backends

        cd ${curr_dir}

    )
}

do_install
