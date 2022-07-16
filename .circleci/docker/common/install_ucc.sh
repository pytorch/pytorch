#!/bin/bash

set -ex

function install_ucx() {
  set -ex
  git clone --recursive https://github.com/openucx/ucx.git
  pushd ucx
  git checkout ${UCX_COMMIT}
  git submodule update --init --recursive

  ./autogen.sh
  ./configure --prefix=$UCX_HOME      \
      --enable-mt                     \
      --enable-profiling              \
      --with-cuda=no                  \
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
  ./configure --prefix=$UCC_HOME --with-ucx=$UCX_HOME --with-nccl=no
  time make -j
  sudo make install

  popd
  rm -rf ucc
}

install_ucx
install_ucc
