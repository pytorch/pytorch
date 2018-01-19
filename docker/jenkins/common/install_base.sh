#!/bin/bash

set -ex

install_ubuntu() {
  # Use AWS mirror if running in EC2
  if [ -n "${EC2:-}" ]; then
    A="archive.ubuntu.com"
    B="us-east-1.ec2.archive.ubuntu.com"
    perl -pi -e "s/${A}/${B}/g" /etc/apt/sources.list
  fi

  apt-get update
  apt-get install -y --no-install-recommends \
          autoconf \
          build-essential \
          ca-certificates \
          cmake \
          curl \
          doxygen \
          git \
          graphviz \
          libgoogle-glog-dev \
          libhiredis-dev \
          libiomp-dev \
          libleveldb-dev \
          liblmdb-dev \
          libopencv-dev \
          libprotobuf-dev \
          libpthread-stubs0-dev \
          libsnappy-dev \
          protobuf-compiler \
          sudo

  # Cleanup
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

install_centos() {
  # Need EPEL for many packages we depend on.
  # See http://fedoraproject.org/wiki/EPEL
  yum --enablerepo=extras install -y epel-release

  # Note: protobuf-c-{compiler,devel} on CentOS are too old to be used
  # for Caffe2. That said, we still install them to make sure the build
  # system opts to build/use protoc and libprotobuf from third-party.
  yum install -y \
      autoconf \
      automake \
      cmake \
      cmake3 \
      curl \
      gcc \
      gcc-c++ \
      gflags-devel \
      git \
      glibc-devel \
      glibc-headers \
      glog-devel \
      hiredis-devel \
      leveldb-devel \
      libstdc++-devel \
      lmdb-devel \
      make \
      opencv-devel \
      protobuf-c-compiler \
      protobuf-c-devel \
      snappy-devel \
      sudo

  # Cleanup
  yum clean all
  rm -rf /var/cache/yum
  rm -rf /var/lib/yum/yumdb
  rm -rf /var/lib/yum/history
}

# Install base packages depending on the base OS
if [ -f /etc/lsb-release ]; then
  install_ubuntu
elif [ -f /etc/os-release ]; then
  install_centos
else
  echo "Unable to determine OS..."
  exit 1
fi
