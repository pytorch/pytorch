#!/usr/bin/env bash
# Script used only in CD pipeline

set -eou pipefail

function do_install() {
    cuda_version=$1
    cuda_version_nodot=${1/./}
    MAGMA_VERSION="2.6.1"
    magma_archive="magma-cuda${cuda_version_nodot}-${MAGMA_VERSION}-1.tar.bz2"
    cuda_dir="/usr/local/cuda-${cuda_version}"

    pushd magma
    make magma-cuda${cuda_version_nodot}

    tar -xvf "${magma_archive}"
    mkdir -p "${cuda_dir}/magma"
    mv include "${cuda_dir}/magma/include"
    mv lib "${cuda_dir}/magma/lib"
    popd
}

do_install $1
