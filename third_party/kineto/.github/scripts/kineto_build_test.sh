#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

GPU_ARCH="${1:?Usage: kineto_build_test.sh <cpu|cuda|rocm> [all|build|test]}"
MODE="${2:-all}"
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ "${MODE}" != "all" && "${MODE}" != "build" && "${MODE}" != "test" ]]; then
  echo "ERROR: unknown mode '${MODE}' (expected all|build|test)" >&2
  exit 1
fi

run_static_tests() {
  pushd build_static
  CTEST_OUTPUT_ON_FAILURE=1 make test
  popd
  echo "====: Ran static libkineto tests"
}

# The split ROCm test job unpacks the static build artifact and only runs
# ctest. setup.sh must already have reinstalled cmake: the generated
# makefiles bake in the absolute ctest path from the build job.
if [[ "${MODE}" == "test" ]]; then
  ARTIFACT_ROOT="${RUNNER_ARTIFACT_DIR:-/artifacts}"
  tar xzf "${ARTIFACT_ROOT}/kineto-build.tar.gz"
  echo "====: Restored build_static from the build job"
  run_static_tests
  exit 0
fi

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

if [[ "${MODE}" == "build" ]]; then
  # Only the static build carries the test binaries; the shared build is
  # here purely as a compile check, so it does not need to be shipped.
  mkdir -p artifacts-to-be-uploaded
  tar czf artifacts-to-be-uploaded/kineto-build.tar.gz build_static
  echo "====: Packed build_static for the test job"
  exit 0
fi

run_static_tests
