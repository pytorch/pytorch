#!/usr/bin/env bash
# Script used only in CD pipeline

set -eou pipefail

function do_install() {
    cuda_version=$1
    cuda_version_nodot=${1/./}

    MAGMA_VERSION="2.9.0"

    cuda_dir="/usr/local/cuda-${cuda_version}"
    (
        set -x
        tmp_dir=$(mktemp -d)
        pushd ${tmp_dir}
        curl -OLs https://icl.utk.edu/projectsfiles/magma/downloads/magma-${MAGMA_VERSION}.tar.gz
        tar -xvf "${magma_archive}"
        mkdir -p "${cuda_dir}/magma"
        mv include "${cuda_dir}/magma/include"
        mv lib "${cuda_dir}/magma/lib"
        popd
    )
}

do_install $1
