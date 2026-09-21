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

# --- Kineto cmake flags ---
# Enable ROCm (roctracer) and disable CUPTI. ROCM_SOURCE_DIR is required by
# CMakeLists.txt to locate roctracer headers and libraries.

# shellcheck disable=SC2034
KINETO_CMAKE_FLAGS=(
  -DKINETO_BACKEND=rocm
  -DROCM_SOURCE_DIR=/opt/rocm
)

# --- PyTorch build environment variables ---

export USE_ROCM=1
export BUILD_TEST=1
export PYTORCH_TEST_WITH_ROCM=1
export PYTORCH_ROCM_ARCH="gfx950"

# Cap parallel compile jobs. PyTorch's build otherwise spawns one compile per
# core, and ROCm never reaches the sccache bucket (see below), so every build
# recompiles the heavy ATen kernels from scratch. Without a cap they compile
# all at once and can exhaust runner memory, tripping the OOM killer.
#
# Note that this is naively set the same as on the CUDA side. We can tweak this
# if needed.
export MAX_JOBS=8

# --- PyTorch build caching ---
# This arch's CI runner is not on AWS and cannot reach PyTorch's S3 sccache
# bucket.
# shellcheck disable=SC2034
KINETO_USE_SCCACHE=0

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
