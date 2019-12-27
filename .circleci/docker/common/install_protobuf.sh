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
  # Ubuntu 14.04 ships with protobuf 2.5, but ONNX needs protobuf >= 2.6
  # so we install that here if on 14.04
  # Ubuntu 14.04 also has cmake 2.8.12 as the default option, so we will
  # install cmake3 here and use cmake3.
  apt-get update
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
  # Centos7 ships with protobuf 2.5, but ONNX needs protobuf >= 2.6
  # so we always install install that here
  install_protobuf_26
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
