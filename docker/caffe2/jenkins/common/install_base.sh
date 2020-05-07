#!/bin/bash

set -ex

# This function installs protobuf 2.6
install_protobuf_26() {
  pb_dir="/usr/temp_pb_install_dir"
  mkdir -p $pb_dir

  # On the nvidia/cuda:9-cudnn7-devel-centos7 image we need this symlink or
  # else it will fail with
  #   g++: error: ./../lib64/crti.o: No such file or directory
  ln -s /usr/lib64 "$pb_dir/lib64"

  curl -LO "https://github.com/google/protobuf/releases/download/v2.6.1/protobuf-2.6.1.tar.gz"
  tar -xvz -C "$pb_dir" --strip-components 1 -f protobuf-2.6.1.tar.gz
  pushd "$pb_dir" && ./configure && make && make check && sudo make install && sudo ldconfig
  popd
  rm -rf $pb_dir
}

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
          apt-transport-https \
          build-essential \
          ca-certificates \
          cmake \
          curl \
          git \
          libgoogle-glog-dev \
          libhiredis-dev \
          libiomp-dev \
          libleveldb-dev \
          liblmdb-dev \
          libopencv-dev \
          libpthread-stubs0-dev \
          libsnappy-dev \
          sudo \
          vim

  # Ubuntu 14.04 ships with protobuf 2.5, but ONNX needs protobuf >= 2.6
  # so we install that here if on 14.04
  # Ubuntu 14.04 also has cmake 2.8.12 as the default option, so we will
  # install cmake3 here and use cmake3.
  if [[ "$UBUNTU_VERSION" == 14.04 ]]; then
    apt-get install -y --no-install-recommends cmake3
    install_protobuf_26
  else
    apt-get install -y --no-install-recommends \
            libprotobuf-dev \
            protobuf-compiler
  fi

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
      snappy-devel \
      sudo

  # Centos7 ships with protobuf 2.5, but ONNX needs protobuf >= 2.6
  # so we always install install that here
  install_protobuf_26

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
