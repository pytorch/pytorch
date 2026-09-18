#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

GPU_ARCH="${1:?Usage: kineto_build_test.sh <cpu|cuda|rocm>}"
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load architecture-specific cmake flags
# shellcheck source=/dev/null
source "${SCRIPTS_DIR}/config_${GPU_ARCH}.sh"

mkdir -p build_static build_shared

pushd build_static
cmake "${KINETO_CMAKE_FLAGS[@]}" -DKINETO_LIBRARY_TYPE=static ../libkineto/
make -j
popd
echo "====: Compiled static libkineto"

pushd build_shared
cmake "${KINETO_CMAKE_FLAGS[@]}" -DKINETO_LIBRARY_TYPE=shared ../libkineto/
make -j
popd
echo "====: Compiled shared libkineto"

pushd build_static
CTEST_OUTPUT_ON_FAILURE=1 make test
popd
echo "====: Ran static libkineto tests"
