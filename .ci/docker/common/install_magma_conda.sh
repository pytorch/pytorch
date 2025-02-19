#!/usr/bin/env bash
# Script that replaces the magma install from a conda package

set -eou pipefail

cuda_version_nodot=${1/./}
anaconda_dir="$CONDA_PREFIX"

MAGMA_VERSION="2.6.1"
magma_archive="magma-cuda${cuda_version_nodot}-${MAGMA_VERSION}-1.tar.bz2"
(
    set -x
    tmp_dir=$(mktemp -d)
    pushd ${tmp_dir}
    curl -OLs https://ossci-linux.s3.us-east-1.amazonaws.com/${magma_archive}
    tar -xvf "${magma_archive}"
    mv include/* "${anaconda_dir}/include/"
    mv lib/* "${anaconda_dir}/lib"
    popd
)
