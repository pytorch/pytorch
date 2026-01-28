#!/bin/bash

set -ex

if [[ -d "/usr/local/cuda/" ]];  then
  with_cuda=/usr/local/cuda/
else
  with_cuda=no
fi

if [[ -f /etc/rocm_env.sh ]]; then
  source /etc/rocm_env.sh
fi

if [[ -d "${ROCM_PATH}" ]]; then
  with_rocm="${ROCM_PATH}"
else
  with_rocm=no
fi

function install_ucx() {
  set -ex
  git clone --recursive https://github.com/openucx/ucx.git
  pushd ucx
  git checkout ${UCX_COMMIT}
  git submodule update --init --recursive

  ./autogen.sh
  ./configure --prefix=$UCX_HOME      \
      --enable-mt                     \
      --with-cuda=$with_cuda          \
      --with-rocm=$with_rocm          \
      --enable-profiling              \
      --enable-stats
  time make -j
  sudo make install

  popd
  rm -rf ucx
}

function install_ucc() {
  set -ex
  git clone --recursive https://github.com/openucx/ucc.git
  pushd ucc
  git checkout ${UCC_COMMIT}
  git submodule update --init --recursive

  ./autogen.sh

  if [[ -n "$CUDA_VERSION"  && $CUDA_VERSION == 13* ]]; then
    NVCC_GENCODE="-gencode=arch=compute_86,code=compute_86"
  else
    # We only run distributed tests on Tesla M60 and A10G
    NVCC_GENCODE="-gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_86,code=compute_86"
  fi

  if [[ -n "$ROCM_VERSION" ]]; then
    if [[ -n "$PYTORCH_ROCM_ARCH" ]]; then
      amdgpu_targets=`echo $PYTORCH_ROCM_ARCH | sed 's/;/ /g'`
    else
      amdgpu_targets=`rocm_agent_enumerator | grep -v gfx000 | sort -u | xargs`
    fi
    for arch in $amdgpu_targets; do
      HIP_OFFLOAD="$HIP_OFFLOAD --offload-arch=$arch"
    done
    HIP_OFFLOAD="$HIP_OFFLOAD --rocm-path=${ROCM_PATH}"

    # Set device library path if detected (handles TheRock vs traditional ROCm)
    if [ -n "${ROCM_DEVICE_LIB_PATH}" ] && [ -d "${ROCM_DEVICE_LIB_PATH}" ]; then
      HIP_OFFLOAD="$HIP_OFFLOAD --rocm-device-lib-path=${ROCM_DEVICE_LIB_PATH}"
    fi
  else
    HIP_OFFLOAD="all-arch-no-native"
  fi

  ./configure --prefix=$UCC_HOME          \
    --with-ucx=$UCX_HOME                  \
    --with-cuda=$with_cuda                \
    --with-nvcc-gencode="${NVCC_GENCODE}" \
    --with-rocm=$with_rocm                \
    --with-rocm-arch="${HIP_OFFLOAD}"
  # First observed by ROCm nightly builds, ucc rccl sources fail compile with
  # error: #warning "NCCL C++ API is disabled because C compiler is being used. [-Werror=cpp]
  # Work-around by adding make CFLAGS=-Wno-error=cpp
  time make -j CFLAGS=-Wno-error=cpp
  sudo make install

  popd
  rm -rf ucc
}

install_ucx
install_ucc
