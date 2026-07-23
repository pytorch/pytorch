#!/usr/bin/env bash
# Script used only in CD pipeline

set -eou pipefail

function do_install() {
    rocm_version=$1
    if [[ ${rocm_version} =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        # chop off any patch version
        rocm_version="${rocm_version%.*}"
    fi

    rocm_version_nodot=${rocm_version//./}

    # https://github.com/icl-utk-edu/magma/pull/77
    MAGMA_VERSION=a68b9257ac435afa1ebdfb8f50d67668950aef61
    magma_archive="magma-rocm${rocm_version_nodot}-${MAGMA_VERSION}-1.tar.bz2"

    rocm_dir="/opt/rocm"
    (
        set -x
        tmp_dir=$(mktemp -d)
        pushd ${tmp_dir}
        curl -OLs https://ossci-linux.s3.us-east-1.amazonaws.com/${magma_archive}
        if tar -xvf "${magma_archive}"
        then
            mkdir -p "${rocm_dir}/magma"
            mv include "${rocm_dir}/magma/include"
            mv lib "${rocm_dir}/magma/lib"
        else
            echo "${magma_archive} not found, skipping magma install"
        fi
        popd
    )
}

do_install $1
