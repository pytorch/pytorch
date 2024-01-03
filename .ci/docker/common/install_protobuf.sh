#!/bin/bash

set -ex

# This function installs protobuf 3.17
install_protobuf_317() {
  pb_dir="/usr/temp_pb_install_dir"
  mkdir -p $pb_dir

  # On the nvidia/cuda:9-cudnn7-devel-centos7 image we need this symlink or
  # else it will fail with
  #   g++: error: ./../lib64/crti.o: No such file or directory
  ln -s /usr/lib64 "$pb_dir/lib64"

  curl -LO "https://github.com/protocolbuffers/protobuf/releases/download/v3.17.3/protobuf-all-3.17.3.tar.gz" --retry 3
  tar -xvz -C "$pb_dir" --strip-components 1 -f protobuf-all-3.17.3.tar.gz
  # -j6 to balance memory usage and speed.
  # naked `-j` seems to use too much memory.
  pushd "$pb_dir" && ./configure && make -j6 && make -j6 check && sudo make -j6 install && sudo ldconfig
  popd
  rm -rf $pb_dir
}

install_ubuntu() {
  # Ubuntu 14.04 has cmake 2.8.12 as the default option, so we will
  # install cmake3 here and use cmake3.
  apt-get update
  if [[ "$UBUNTU_VERSION" == 14.04 ]]; then
    apt-get install -y --no-install-recommends cmake3
  fi

  # Cleanup
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

  install_protobuf_317
}

install_centos() {
  install_protobuf_317
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
