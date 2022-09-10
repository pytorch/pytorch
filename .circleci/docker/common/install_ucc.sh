#!/bin/bash

set -ex

if [[ -d "/usr/local/cuda/" ]];  then
  with_cuda=/usr/local/cuda/
else
  with_cuda=no
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
  ./configure --prefix=$UCC_HOME --with-ucx=$UCX_HOME --with-cuda=$with_cuda
  time make -j
  sudo make install

  popd
  rm -rf ucc
}

install_ucx
install_ucc
