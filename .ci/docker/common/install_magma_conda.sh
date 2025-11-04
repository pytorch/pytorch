#!/usr/bin/env bash
# Script that installs magma from tarball inside conda environment.
# It replaces anaconda magma-cuda package which is no longer published.
# Execute it inside active conda environment.
# See issue: https://github.com/pytorch/pytorch/issues/138506

set -eou pipefail

cuda_version_nodot=${1/./}
anaconda_dir=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

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
