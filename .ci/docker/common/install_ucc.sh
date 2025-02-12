#!/bin/bash

set -ex

if [[ -d "/usr/local/cuda/" ]];  then
  with_cuda=/usr/local/cuda/
else
  with_cuda=no
fi

if [[ -d "/opt/rocm" ]]; then
  with_rocm=/opt/rocm
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

  # We only run distributed tests on Tesla M60 and A10G
  NVCC_GENCODE="-gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_86,code=compute_86"

  if [[ -n "$ROCM_VERSION" ]]; then
    if [[ -n "$PYTORCH_ROCM_ARCH" ]]; then
      amdgpu_targets=`echo $PYTORCH_ROCM_ARCH | sed 's/;/ /g'`
    else
      amdgpu_targets=`rocm_agent_enumerator | grep -v gfx000 | sort -u | xargs`
    fi
    for arch in $amdgpu_targets; do
      HIP_OFFLOAD="$HIP_OFFLOAD --offload-arch=$arch"
    done
  else
    HIP_OFFLOAD="all-arch-no-native"
  fi

  ./configure --prefix=$UCC_HOME          \
    --with-ucx=$UCX_HOME                  \
    --with-cuda=$with_cuda                \
    --with-nvcc-gencode="${NVCC_GENCODE}" \
    --with-rocm=$with_rocm                \
    --with-rocm-arch="${HIP_OFFLOAD}"
  time make -j
  sudo make install

  popd
  rm -rf ucc
}

install_ucx
install_ucc
