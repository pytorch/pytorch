#!/bin/bash

set -ex

function install_nvpl {

    mkdir -p /opt/nvpl/lib /opt/nvpl/include

    wget https://developer.download.nvidia.com/compute/nvpl/redist/nvpl_blas/linux-sbsa/nvpl_blas-linux-sbsa-0.5.0.1-archive.tar.xz
    tar xf nvpl_blas-linux-sbsa-0.5.0.1-archive.tar.xz
    cp -r nvpl_blas-linux-sbsa-0.5.0.1-archive/lib/* /opt/nvpl/lib/
    cp -r nvpl_blas-linux-sbsa-0.5.0.1-archive/include/* /opt/nvpl/include/

    wget https://developer.download.nvidia.com/compute/nvpl/redist/nvpl_lapack/linux-sbsa/nvpl_lapack-linux-sbsa-0.3.2-archive.tar.xz
    tar xf nvpl_lapack-linux-sbsa-0.3.2-archive.tar.xz
    cp -r nvpl_lapack-linux-sbsa-0.3.2-archive/lib/* /opt/nvpl/lib/
    cp -r nvpl_lapack-linux-sbsa-0.3.2-archive/include/* /opt/nvpl/include/
}

install_nvpl