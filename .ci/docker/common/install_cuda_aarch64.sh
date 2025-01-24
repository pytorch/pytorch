#!/bin/bash
# Script used only in CD pipeline

set -ex

NCCL_VERSION=v2.21.5-1
CUDNN_VERSION=9.5.1.17

function install_cusparselt_062 {
    # cuSparseLt license: https://docs.nvidia.com/cuda/cusparselt/license.html
    mkdir tmp_cusparselt && pushd tmp_cusparselt
    wget -q https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-sbsa/libcusparse_lt-linux-sbsa-0.6.2.3-archive.tar.xz
    tar xf libcusparse_lt-linux-sbsa-0.6.2.3-archive.tar.xz
    cp -a libcusparse_lt-linux-sbsa-0.6.2.3-archive/include/* /usr/local/cuda/include/
    cp -a libcusparse_lt-linux-sbsa-0.6.2.3-archive/lib/* /usr/local/cuda/lib64/
    popd
    rm -rf tmp_cusparselt
}

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

function install_124 {
  CUDNN_VERSION=9.1.0.70
  echo "Installing CUDA 12.4.1 and cuDNN ${CUDNN_VERSION} and NCCL ${NCCL_VERSION} and cuSparseLt-0.6.2"
  rm -rf /usr/local/cuda-12.4 /usr/local/cuda
  # install CUDA 12.4.1 in the same container
  wget -q https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux_sbsa.run
  chmod +x cuda_12.4.1_550.54.15_linux_sbsa.run
  ./cuda_12.4.1_550.54.15_linux_sbsa.run --toolkit --silent
  rm -f cuda_12.4.1_550.54.15_linux_sbsa.run
  rm -f /usr/local/cuda && ln -s /usr/local/cuda-12.4 /usr/local/cuda

  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  mkdir tmp_cudnn && cd tmp_cudnn
  wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-sbsa/cudnn-linux-sbsa-${CUDNN_VERSION}_cuda12-archive.tar.xz -O cudnn-linux-sbsa-${CUDNN_VERSION}_cuda12-archive.tar.xz
  tar xf cudnn-linux-sbsa-${CUDNN_VERSION}_cuda12-archive.tar.xz
  cp -a cudnn-linux-sbsa-${CUDNN_VERSION}_cuda12-archive/include/* /usr/local/cuda/include/
  cp -a cudnn-linux-sbsa-${CUDNN_VERSION}_cuda12-archive/lib/* /usr/local/cuda/lib64/
  cd ..
  rm -rf tmp_cudnn

  # NCCL license: https://docs.nvidia.com/deeplearning/nccl/#licenses
  # Follow build: https://github.com/NVIDIA/nccl/tree/master?tab=readme-ov-file#build
  git clone -b ${NCCL_VERSION} --depth 1 https://github.com/NVIDIA/nccl.git
  cd nccl && make -j src.build
  cp -a build/include/* /usr/local/cuda/include/
  cp -a build/lib/* /usr/local/cuda/lib64/
  cd ..
  rm -rf nccl

  install_cusparselt_062

  ldconfig
}

function prune_124 {
  echo "Pruning CUDA 12.4"
  #####################################################################################
  # CUDA 12.4 prune static libs
  #####################################################################################
  export NVPRUNE="/usr/local/cuda-12.4/bin/nvprune"
  export CUDA_LIB_DIR="/usr/local/cuda-12.4/lib64"

  export GENCODE="-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"
  export GENCODE_CUDNN="-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"

  if [[ -n "$OVERRIDE_GENCODE" ]]; then
      export GENCODE=$OVERRIDE_GENCODE
  fi

  # all CUDA libs except CuDNN and CuBLAS
  ls $CUDA_LIB_DIR/ | grep "\.a" | grep -v "culibos" | grep -v "cudart" | grep -v "cudnn" | grep -v "cublas" | grep -v "metis"  \
      | xargs -I {} bash -c \
                "echo {} && $NVPRUNE $GENCODE $CUDA_LIB_DIR/{} -o $CUDA_LIB_DIR/{}"

  # prune CuDNN and CuBLAS
  $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublas_static.a -o $CUDA_LIB_DIR/libcublas_static.a
  $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublasLt_static.a -o $CUDA_LIB_DIR/libcublasLt_static.a

  #####################################################################################
  # CUDA 12.4 prune visual tools
  #####################################################################################
  export CUDA_BASE="/usr/local/cuda-12.4/"
  rm -rf $CUDA_BASE/libnvvp $CUDA_BASE/nsightee_plugins $CUDA_BASE/nsight-compute-2024.1.0 $CUDA_BASE/nsight-systems-2023.4.4/
}

function install_126 {
  echo "Installing CUDA 12.6.3 and cuDNN ${CUDNN_VERSION} and NCCL ${NCCL_VERSION} and cuSparseLt-0.6.3"
  rm -rf /usr/local/cuda-12.6 /usr/local/cuda
  # install CUDA 12.6.3 in the same container
  wget -q https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux_sbsa.run
  chmod +x cuda_12.6.3_560.35.05_linux_sbsa.run
  ./cuda_12.6.3_560.35.05_linux_sbsa.run --toolkit --silent
  rm -f cuda_12.6.3_560.35.05_linux_sbsa.run
  rm -f /usr/local/cuda && ln -s /usr/local/cuda-12.6 /usr/local/cuda

  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  mkdir tmp_cudnn && cd tmp_cudnn
  wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-sbsa/cudnn-linux-sbsa-${CUDNN_VERSION}_cuda12-archive.tar.xz -O cudnn-linux-sbsa-${CUDNN_VERSION}_cuda12-archive.tar.xz
  tar xf cudnn-linux-sbsa-${CUDNN_VERSION}_cuda12-archive.tar.xz
  cp -a cudnn-linux-sbsa-${CUDNN_VERSION}_cuda12-archive/include/* /usr/local/cuda/include/
  cp -a cudnn-linux-sbsa-${CUDNN_VERSION}_cuda12-archive/lib/* /usr/local/cuda/lib64/
  cd ..
  rm -rf tmp_cudnn

  # NCCL license: https://docs.nvidia.com/deeplearning/nccl/#licenses
  # Follow build: https://github.com/NVIDIA/nccl/tree/master?tab=readme-ov-file#build
  git clone -b ${NCCL_VERSION} --depth 1 https://github.com/NVIDIA/nccl.git
  cd nccl && make -j src.build
  cp -a build/include/* /usr/local/cuda/include/
  cp -a build/lib/* /usr/local/cuda/lib64/
  cd ..
  rm -rf nccl

  install_cusparselt_063

  ldconfig
}

function prune_126 {
  echo "Pruning CUDA 12.6"
  #####################################################################################
  # CUDA 12.6 prune static libs
  #####################################################################################
  export NVPRUNE="/usr/local/cuda-12.6/bin/nvprune"
  export CUDA_LIB_DIR="/usr/local/cuda-12.6/lib64"

  export GENCODE="-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"
  export GENCODE_CUDNN="-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"

  if [[ -n "$OVERRIDE_GENCODE" ]]; then
      export GENCODE=$OVERRIDE_GENCODE
  fi
  if [[ -n "$OVERRIDE_GENCODE_CUDNN" ]]; then
      export GENCODE_CUDNN=$OVERRIDE_GENCODE_CUDNN
  fi

  # all CUDA libs except CuDNN and CuBLAS
  ls $CUDA_LIB_DIR/ | grep "\.a" | grep -v "culibos" | grep -v "cudart" | grep -v "cudnn" | grep -v "cublas" | grep -v "metis"  \
      | xargs -I {} bash -c \
                "echo {} && $NVPRUNE $GENCODE $CUDA_LIB_DIR/{} -o $CUDA_LIB_DIR/{}"

  # prune CuDNN and CuBLAS
  $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublas_static.a -o $CUDA_LIB_DIR/libcublas_static.a
  $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublasLt_static.a -o $CUDA_LIB_DIR/libcublasLt_static.a

  #####################################################################################
  # CUDA 12.6 prune visual tools
  #####################################################################################
  export CUDA_BASE="/usr/local/cuda-12.6/"
  rm -rf $CUDA_BASE/libnvvp $CUDA_BASE/nsightee_plugins $CUDA_BASE/nsight-compute-2024.3.2 $CUDA_BASE/nsight-systems-2024.5.1/
}

# idiomatic parameter and option handling in sh
while test $# -gt 0
do
    case "$1" in
    12.4) install_124; prune_124
        ;;
    12.6) install_126; prune_126
        ;;
    *) echo "bad argument $1"; exit 1
        ;;
    esac
    shift
done
