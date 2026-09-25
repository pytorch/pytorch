#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Architecture-specific configuration for ROCm CI.
#
# This file is sourced by kineto_build_test.sh and pytorch_build_test.sh.
# It defines:
#   - Extra cmake flags for the libkineto build
#   - Environment variables for the PyTorch build
#   - Deselected pytest tests
#

# --- Detect ROCM_SOURCE_DIR and ROCM_INCLUDE_DIRS ---
# ROCm ships across two pip wheels (_rocm_sdk_core, _rocm_sdk_devel) but the
# split of headers vs libraries varies by wheel version. Probe each package
# for what it actually contains rather than assuming.
if [ -z "${ROCM_SOURCE_DIR:-}" ]; then
  _rocm_lib_path=$(python -c "
import importlib.util, pathlib
for pkg in ('_rocm_sdk_core', '_rocm_sdk_devel'):
    spec = importlib.util.find_spec(pkg)
    if spec:
        p = pathlib.Path(spec.submodule_search_locations[0])
        if (p / 'lib' / 'librocprofiler-sdk.so').exists():
            print(str(p))
            break
" 2>/dev/null || true)
  _rocm_inc_path=$(python -c "
import importlib.util, pathlib
for pkg in ('_rocm_sdk_devel', '_rocm_sdk_core'):
    spec = importlib.util.find_spec(pkg)
    if spec:
        p = pathlib.Path(spec.submodule_search_locations[0])
        if (p / 'include' / 'rocm-core' / 'rocm_version.h').exists():
            print(str(p))
            break
" 2>/dev/null || true)
  if [ -z "${_rocm_lib_path}" ]; then
    echo "ERROR: librocprofiler-sdk.so not found in any ROCm pip wheel."
    exit 1
  fi
  if [ -z "${_rocm_inc_path}" ]; then
    echo "ERROR: rocm-core/rocm_version.h not found in any ROCm pip wheel."
    exit 1
  fi
  export ROCM_SOURCE_DIR="${_rocm_lib_path}"
  echo "====: ROCM_SOURCE_DIR (runtime libs): ${ROCM_SOURCE_DIR}"
  if [ "${_rocm_inc_path}" != "${_rocm_lib_path}" ]; then
    export ROCM_INCLUDE_DIRS="${_rocm_inc_path}/include"
    echo "====: ROCM_INCLUDE_DIRS (headers): ${ROCM_INCLUDE_DIRS}"
  fi
fi

# --- Kineto cmake flags ---
# Enable ROCm (rocprofiler-sdk). ROCM_SOURCE_DIR is required by CMakeLists.txt
# to locate headers and libraries.

# shellcheck disable=SC2034
KINETO_CMAKE_FLAGS=(
  -DKINETO_BACKEND=rocm
  -DROCM_SOURCE_DIR="${ROCM_SOURCE_DIR}"
)
if [ -n "${ROCM_INCLUDE_DIRS:-}" ]; then
  KINETO_CMAKE_FLAGS+=(-DROCM_INCLUDE_DIRS="${ROCM_INCLUDE_DIRS}")
fi

# --- PyTorch build environment variables ---

export USE_ROCM=1
export BUILD_TEST=1
export PYTORCH_TEST_WITH_ROCM=1
export PYTORCH_ROCM_ARCH="gfx950"

# Cap parallel compile jobs. PyTorch's build otherwise spawns one compile per
# core. Without a cap they compile all at once and can exhaust runner memory,
# tripping the OOM killer.
#
# Note that this is naively set the same as on the CUDA side. We can tweak this
# if needed.
export MAX_JOBS=8

# --- PyTorch build caching ---
# The ROCm wheel is built on linux.12xlarge (AWS), which can reach PyTorch's
# shared S3 sccache bucket. The MI350 test job never compiles.
# shellcheck disable=SC2034
KINETO_USE_SCCACHE=1

# --- Deselected PyTorch profiler tests ---
# Each entry is a pytest node ID passed as a --deselect argument.
#
# Dynamic skipping of known-broken/flaky upstream tests is handled via
# DISABLED_TESTS_FILE in pytorch_build_test.sh. The hardcoded list below
# supplements it for tests not yet tracked upstream.

# shellcheck disable=SC2034
DESELECTED_TESTS=(
  test/profiler/test_profiler.py::TestExperimentalUtils::test_fuzz_symbolize
)
