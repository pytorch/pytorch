#!/bin/bash

function install_ucx() {
  git clone --recursive https://github.com/openucx/ucx.git
  pushd ucx
  git checkout ${UCX_COMMIT}
  ./autogen.sh
  ./configure --prefix=$UCX_HOME      \
      --enable-mt                     \
      --enable-profiling              \
      --enable-stats
  time make -j
  sudo make install
  popd
}

function install_ucc() {
  rm -rf ucc
  git clone --recursive https://github.com/openucx/ucc.git
  pushd ucc
  git checkout ${UCC_COMMIT}
  ./autogen.sh
  ./configure --prefix=$UCC_HOME --with-ucx=$UCX_HOME
  time make -j
  sudo make install
  popd
}

install_ucx
install_ucc
