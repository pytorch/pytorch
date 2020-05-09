#!/bin/bash

set -ex

if [[ "$UBUNTU_VERSION" == "14.04" ]]; then
  # cmake 2 is too old
  cmake3=cmake3
else
  cmake3=cmake
fi

if [[ "$UBUNTU_VERSION" == "18.04" ]]; then
  cmake3="cmake=3.10*"
else
  cmake3="${cmake3}=3.5*"
fi

# Install common dependencies
apt-get update
# TODO: Some of these may not be necessary
# TODO: libiomp also gets installed by conda, aka there's a conflict
ccache_deps="asciidoc docbook-xml docbook-xsl xsltproc"
numpy_deps="gfortran"
apt-get install -y --no-install-recommends \
  $ccache_deps \
  $numpy_deps \
  ${cmake3} \
  apt-transport-https \
  autoconf \
  automake \
  build-essential \
  ca-certificates \
  curl \
  git \
  libatlas-base-dev \
  libc6-dbg \
  libiomp-dev \
  libyaml-dev \
  libz-dev \
  libjpeg-dev \
  libasound2-dev \
  libsndfile-dev \
  python \
  python-dev \
  python-setuptools \
  python-wheel \
  software-properties-common \
  sudo \
  wget \
  vim

# Install Valgrind separately since the apt-get version is too old.
mkdir valgrind_build && cd valgrind_build
if ! wget http://valgrind.org/downloads/valgrind-3.14.0.tar.bz2
then
  wget https://sourceware.org/ftp/valgrind/valgrind-3.14.0.tar.bz2
fi
tar -xjf valgrind-3.14.0.tar.bz2
cd valgrind-3.14.0
./configure --prefix=/usr/local
make
sudo make install
cd ../../
rm -rf valgrind_build
alias valgrind="/usr/local/bin/valgrind"

# TODO: THIS IS A HACK!!!
# distributed nccl(2) tests are a bit busted, see https://github.com/pytorch/pytorch/issues/5877
if dpkg -s libnccl-dev; then
  apt-get remove -y libnccl-dev libnccl2 --allow-change-held-packages
fi

# Cleanup package manager
apt-get autoclean && apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
