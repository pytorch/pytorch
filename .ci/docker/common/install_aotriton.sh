#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

TARBALL='aotriton.tar.bz2'
# This read command alwasy returns with exit code 1
read -d "\n" VER MANYLINUX ROCMBASE PINNED_COMMIT SHA256 < aotriton_version.txt || true
ARCH=$(uname -m)
AOTRITON_INSTALL_PREFIX="$1"
AOTRITON_URL="https://github.com/ROCm/aotriton/releases/download/${VER}/aotriton-${VER}-${MANYLINUX}_${ARCH}-${ROCMBASE}.tar.bz2"

cd "${AOTRITON_INSTALL_PREFIX}"
# Must use -L to follow redirects
curl -L --retry 3 -o "${TARBALL}" "${AOTRITON_URL}"
ACTUAL_SHA256=$(sha256sum "${TARBALL}" | cut -d " " -f 1)
if [ "${SHA256}" != "${ACTUAL_SHA256}" ]; then
  echo -n "Error: The SHA256 of downloaded tarball is ${ACTUAL_SHA256},"
  echo " which does not match the expected value ${SHA256}."
  exit
fi
tar xf "${TARBALL}" && rm -rf "${TARBALL}"
