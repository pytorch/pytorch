#!/bin/bash

set -ex

arch_path=''
targetarch=${TARGETARCH:-$(uname -m)}
if [ ${targetarch} = 'amd64' ] || [ "${targetarch}" = 'x86_64' ]; then
  arch_path='x86_64'
else
  arch_path='sbsa'
fi

function install_cuda {
  version=$1
  runfile=$2
  major_minor=${version%.*}
  rm -rf /usr/local/cuda-${major_minor} /usr/local/cuda
  if [[ ${arch_path} == 'sbsa' ]]; then
      runfile="${runfile}_sbsa"
  fi
  runfile="${runfile}.run"
  wget -q https://developer.download.nvidia.com/compute/cuda/${version}/local_installers/${runfile} -O ${runfile}
  chmod +x ${runfile}
  ./${runfile} --toolkit --silent
  rm -f ${runfile}
  rm -f /usr/local/cuda && ln -s /usr/local/cuda-${major_minor} /usr/local/cuda
}

function install_cudnn {
  cuda_major_version=$1
  cudnn_version=$2
  mkdir tmp_cudnn && cd tmp_cudnn
  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  filepath="cudnn-linux-${arch_path}-${cudnn_version}_cuda${cuda_major_version}-archive"
  wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-${arch_path}/${filepath}.tar.xz
  tar xf ${filepath}.tar.xz
  cp -a ${filepath}/include/* /usr/local/cuda/include/
  cp -a ${filepath}/lib/* /usr/local/cuda/lib64/
  cd ..
  rm -rf tmp_cudnn
}

function install_118 {
    CUDNN_VERSION=9.1.0.70
    echo "Installing CUDA 11.8 and cuDNN ${CUDNN_VERSION} and NCCL and cuSparseLt-0.4.0"
    install_cuda 11.8.0 cuda_11.8.0_520.61.05_linux

    install_cudnn 11 $CUDNN_VERSION

    CUDA_VERSION=11.8 bash install_nccl.sh

    CUDA_VERSION=11.8 bash install_cusparselt.sh

    ldconfig
}

function install_124 {
  CUDNN_VERSION=9.1.0.70
  echo "Installing CUDA 12.4.1 and cuDNN ${CUDNN_VERSION} and NCCL and cuSparseLt-0.6.2"
  install_cuda 12.4.1 cuda_12.4.1_550.54.15_linux

  install_cudnn 12 $CUDNN_VERSION

  CUDA_VERSION=12.4 bash install_nccl.sh

  CUDA_VERSION=12.4 bash install_cusparselt.sh

  ldconfig
}

function install_126 {
  CUDNN_VERSION=9.5.1.17
  echo "Installing CUDA 12.6.3 and cuDNN ${CUDNN_VERSION} and NCCL and cuSparseLt-0.6.3"
  install_cuda 12.6.3 cuda_12.6.3_560.35.05_linux

  install_cudnn 12 $CUDNN_VERSION

  CUDA_VERSION=12.6 bash install_nccl.sh

  CUDA_VERSION=12.6 bash install_cusparselt.sh

  ldconfig
}

function prune_118 {
    echo "Pruning CUDA 11.8 and cuDNN"
    #####################################################################################
    # CUDA 11.8 prune static libs
    #####################################################################################
    export NVPRUNE="/usr/local/cuda-11.8/bin/nvprune"
    export CUDA_LIB_DIR="/usr/local/cuda-11.8/lib64"

    export GENCODE="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"
    export GENCODE_CUDNN="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"

    if [[ -n "$OVERRIDE_GENCODE" ]]; then
        export GENCODE=$OVERRIDE_GENCODE
    fi

    # all CUDA libs except CuDNN and CuBLAS (cudnn and cublas need arch 3.7 included)
    ls $CUDA_LIB_DIR/ | grep "\.a" | grep -v "culibos" | grep -v "cudart" | grep -v "cudnn" | grep -v "cublas" | grep -v "metis"  \
      | xargs -I {} bash -c \
                "echo {} && $NVPRUNE $GENCODE $CUDA_LIB_DIR/{} -o $CUDA_LIB_DIR/{}"

    # prune CuDNN and CuBLAS
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublas_static.a -o $CUDA_LIB_DIR/libcublas_static.a
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublasLt_static.a -o $CUDA_LIB_DIR/libcublasLt_static.a

    #####################################################################################
    # CUDA 11.8 prune visual tools
    #####################################################################################
    export CUDA_BASE="/usr/local/cuda-11.8/"
    rm -rf $CUDA_BASE/libnvvp $CUDA_BASE/nsightee_plugins $CUDA_BASE/nsight-compute-2022.3.0 $CUDA_BASE/nsight-systems-2022.4.2/
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
  # CUDA 12.4 prune visual tools
  #####################################################################################
  export CUDA_BASE="/usr/local/cuda-12.4/"
  rm -rf $CUDA_BASE/libnvvp $CUDA_BASE/nsightee_plugins $CUDA_BASE/nsight-compute-2024.1.0 $CUDA_BASE/nsight-systems-2023.4.4/
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

function install_128 {
  CUDNN_VERSION=9.8.0.87
  echo "Installing CUDA 12.8.0 and cuDNN ${CUDNN_VERSION} and NCCL and cuSparseLt-0.6.3"
  # install CUDA 12.8.0 in the same container
  install_cuda 12.8.0 cuda_12.8.0_570.86.10_linux

  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  install_cudnn 12 $CUDNN_VERSION

  CUDA_VERSION=12.8 bash install_nccl.sh

  CUDA_VERSION=12.8 bash install_cusparselt.sh

  ldconfig
}

# idiomatic parameter and option handling in sh
while test $# -gt 0
do
    case "$1" in
    11.8) install_118; prune_118
        ;;
    12.4) install_124; prune_124
        ;;
    12.6) install_126; prune_126
        ;;
    12.8) install_128;
        ;;
    *) echo "bad argument $1"; exit 1
        ;;
    esac
    shift
done
