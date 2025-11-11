#!/usr/bin/env bash
# Script used only in CD pipeline to build and install rocSHMEM

set -eou pipefail

function do_install() {
    ROCSHMEM_VERSION=cba2e3de2fd38b61ff25c34e60d4207b1475aafc
    rocm_dir="/opt/rocm"
    (
        set -x
        curr_dir=$(pwd)
        tmp_dir=$(mktemp -d)

        sudo apt update -y && sudo apt install -y libibverbs-dev

        git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git
        cd rocm-systems
        git sparse-checkout set --cone projects/rocshmem
        git checkout ${ROCSHMEM_VERSION}

        cd projects/rocshmem
        mkdir build
        cd build
        INSTALL_PREFIX=${rocm_dir} ../scripts/build_configs/all_backends

        cd ${curr_dir}

    )
}

do_install
