#!/bin/bash

set -ex

[ -n "$CMAKE_VERSION" ]

# Remove system cmake install so it won't get used instead
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    apt-get remove cmake -y
    ;;
  centos)
    yum remove cmake -y
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac

# Turn 3.6.3 into v3.6
path=$(echo "${CMAKE_VERSION}" | sed -e 's/\([0-9].[0-9]\+\).*/v\1/')
file="cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz"

# Download and install specific CMake version in /usr/local
pushd /tmp
curl -Os --retry 3 "https://cmake.org/files/${path}/${file}"
tar -C /usr/local --strip-components 1 --no-same-owner -zxf cmake-*.tar.gz
rm -f cmake-*.tar.gz
popd
