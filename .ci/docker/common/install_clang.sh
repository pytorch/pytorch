#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

install_ubuntu() {
  apt-get update

  apt-get install -y --no-install-recommends clang-"$CLANG_VERSION"
  apt-get install -y --no-install-recommends llvm-"$CLANG_VERSION"
  # Also require LLD linker from llvm and libomp to build PyTorch from source
  apt-get install -y lld "libomp-${CLANG_VERSION}-dev" "libc++-${CLANG_VERSION}-dev"

  # Use update-alternatives to make this version the default
  update-alternatives --install /usr/bin/clang clang /usr/bin/clang-"$CLANG_VERSION" 50
  update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-"$CLANG_VERSION" 50
  # Override cc/c++ to clang as well
  update-alternatives --install /usr/bin/cc cc /usr/bin/clang 50
  update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 50

  # Cleanup package manager
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

if [ -n "$CLANG_VERSION" ]; then
  # Install base packages depending on the base OS
  ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
  case "$ID" in
    ubuntu)
      install_ubuntu
      ;;
    *)
      echo "Unable to determine OS..."
      exit 1
      ;;
  esac
fi
