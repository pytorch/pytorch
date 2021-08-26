#!/bin/bash

set -ex

export UCX_HOME="/usr"
export UCC_HOME="/usr"

cd /tmp
git clone https://github.com/openucx/ucx.git
git clone https://github.com/openucx/ucc.git

pushd ucx
./autogen.sh
# ./contrib/configure-release-mt --prefix=$UCX_HOME --with-cuda=/usr/local/cuda/
./contrib/configure --enable-mt --prefix=$UCX_HOME --with-cuda=/usr/local/cuda/
make -j
make install
popd

pushd ucc
./autogen.sh
./configure --prefix=$UCC_HOME              \
    --with-ucx=$UCX_HOME                    \
    --with-cuda=/usr/local/cuda/
gcc --version
make -j
make install
popd

rm -rf ucx ucc
