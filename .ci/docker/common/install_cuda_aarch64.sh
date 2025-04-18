#!/bin/bash
# Script used only in CD pipeline

set -ex

CUDNN_VERSION=9.8.0.87

function install_cusparselt_063 {
    # cuSparseLt license: https://docs.nvidia.com/cuda/cusparselt/license.html
    mkdir tmp_cusparselt && pushd tmp_cusparselt
    wget -q https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-sbsa/libcusparse_lt-linux-sbsa-0.6.3.2-archive.tar.xz
    tar xf libcusparse_lt-linux-sbsa-0.6.3.2-archive.tar.xz
    cp -a libcusparse_lt-linux-sbsa-0.6.3.2-archive/include/* /usr/local/cuda/include/
    cp -a libcusparse_lt-linux-sbsa-0.6.3.2-archive/lib/* /usr/local/cuda/lib64/
    popd
    rm -rf tmp_cusparselt
}

function install_128 {
  echo "Installing CUDA 12.8.0 and cuDNN ${CUDNN_VERSION} and NCCL and cuSparseLt-0.6.3"
  rm -rf /usr/local/cuda-12.8 /usr/local/cuda
  # install CUDA 12.8.0 in the same container
  wget -q https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux_sbsa.run
  chmod +x cuda_12.8.0_570.86.10_linux_sbsa.run
  ./cuda_12.8.0_570.86.10_linux_sbsa.run --toolkit --silent
  rm -f cuda_12.8.0_570.86.10_linux_sbsa.run
  rm -f /usr/local/cuda && ln -s /usr/local/cuda-12.8 /usr/local/cuda

  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  mkdir tmp_cudnn && cd tmp_cudnn
  wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-sbsa/cudnn-linux-sbsa-${CUDNN_VERSION}_cuda12-archive.tar.xz -O cudnn-linux-sbsa-${CUDNN_VERSION}_cuda12-archive.tar.xz
  tar xf cudnn-linux-sbsa-${CUDNN_VERSION}_cuda12-archive.tar.xz
  cp -a cudnn-linux-sbsa-${CUDNN_VERSION}_cuda12-archive/include/* /usr/local/cuda/include/
  cp -a cudnn-linux-sbsa-${CUDNN_VERSION}_cuda12-archive/lib/* /usr/local/cuda/lib64/
  cd ..
  rm -rf tmp_cudnn

  CUDA_VERSION=12.8 bash install_nccl.sh

  install_cusparselt_063

  ldconfig
}

# idiomatic parameter and option handling in sh
while test $# -gt 0
do
    case "$1" in
    12.8) install_128;
        ;;
    *) echo "bad argument $1"; exit 1
        ;;
    esac
    shift
done
