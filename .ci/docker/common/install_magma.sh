#!/usr/bin/env bash
# Script used only in CD pipeline

set -eou pipefail

function do_install() {
    cuda_version=$1
    cuda_version_nodot=${1/./}

    # Temporary WAR to be updated for CUDA 12.8
    if [ "$cuda_version_nodot" == "128" ]; then
        # Set it to 12.6 if it matches
        cuda_version_nodot="126"
    fi

    MAGMA_VERSION="2.6.1"
    magma_archive="magma-cuda${cuda_version_nodot}-${MAGMA_VERSION}-1.tar.bz2"

    cuda_dir="/usr/local/cuda-${cuda_version}"
    (
        set -x
        tmp_dir=$(mktemp -d)
        pushd ${tmp_dir}
        curl -OLs https://ossci-linux.s3.us-east-1.amazonaws.com/${magma_archive}
        tar -xvf "${magma_archive}"
        mkdir -p "${cuda_dir}/magma"
        mv include "${cuda_dir}/magma/include"
        mv lib "${cuda_dir}/magma/lib"
        popd
    )
}

do_install $1
