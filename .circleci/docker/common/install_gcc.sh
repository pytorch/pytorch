#!/bin/bash

set -ex

if [ -n "$GCC_VERSION" ]; then

  # Need the official toolchain repo to get alternate packages
  add-apt-repository ppa:ubuntu-toolchain-r/test
  apt-get update
  if [ "$UBUNTU_VERSION" = "16.04" -a "$GCC_VERSION" = "5" ]; then
    apt-get install -y g++-5=5.4.0-6ubuntu1~16.04.12
  else
    apt-get install -y g++-$GCC_VERSION
  fi

  update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-"$GCC_VERSION" 50
  update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-"$GCC_VERSION" 50
  update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-"$GCC_VERSION" 50

  # Cleanup package manager
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

fi
