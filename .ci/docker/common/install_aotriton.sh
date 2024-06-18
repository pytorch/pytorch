#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

AOTRITON_DIR="aotriton"
AOTRITON_PINNED_NAME="aotriton" # No .txt extension
AOTRITON_PINNED_COMMIT=$(get_pinned_commit ${AOTRITON_PINNED_NAME})
AOTRITON_INSTALL_PREFIX="$1"

git clone https://github.com/ROCm/aotriton.git "${AOTRITON_DIR}"
cd "${AOTRITON_DIR}"
git checkout "${AOTRITON_PINNED_COMMIT}"
git submodule sync --recursive
git submodule update --init --recursive --force --depth 1
mkdir build
cd build
cmake .. -G Ninja -DCMAKE_INSTALL_PREFIX=./install_dir -DCMAKE_BUILD_TYPE=Release -DAOTRITON_COMPRESS_KERNEL=OFF -DAOTRITON_NO_PYTHON=ON -DAOTRITON_NO_SHARED=ON
ninja install
mkdir -p "${AOTRITON_INSTALL_PREFIX}"
cp -r install_dir/* "${AOTRITON_INSTALL_PREFIX}"
find /tmp/ -mindepth 1 -delete
rm -rf ~/.triton
