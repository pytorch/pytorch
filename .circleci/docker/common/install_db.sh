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
  apt-get update
  apt-get install -y --no-install-recommends \
          libhiredis-dev \
          libleveldb-dev \
          liblmdb-dev \
          libsnappy-dev

  # Cleanup
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

install_centos() {
  # Need EPEL for many packages we depend on.
  # See http://fedoraproject.org/wiki/EPEL
  yum --enablerepo=extras install -y epel-release

  yum install -y \
      hiredis-devel \
      leveldb-devel \
      lmdb-devel \
      snappy-devel

  # Cleanup
  yum clean all
  rm -rf /var/cache/yum
  rm -rf /var/lib/yum/yumdb
  rm -rf /var/lib/yum/history
}

# Install base packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    install_ubuntu
    ;;
  centos)
    install_centos
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac
