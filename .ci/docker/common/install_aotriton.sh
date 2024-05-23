#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

AOTRITON_DIR="aotriton"
AOTRITON_TEXT_FILE="aotriton.txt"
AOTRITON_PINNED_COMMIT=$(get_pinned_commit ${AOTRITON_TEXT_FILE})
AOTRITON_INSTALL_PREFIX="$1"

git clone https://github.com/ROCm/aotriton.git "${AOTRITON_DIR}"
cd "${AOTRITON_DIR}"
git checkout "${AOTRITON_PINNED_COMMIT}"
git submodule sync --recursive
git submodule update --init --recursive --force --depth 1
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=./install_dir -DCMAKE_BUILD_TYPE=Release -G Ninja
ninja install
SUDO mkdir -p "${AOTRITON_INSTALL_PREFIX}"
SUDO cp -r install_dir/* "${AOTRITON_INSTALL_PREFIX}"
