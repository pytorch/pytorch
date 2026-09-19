#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Architecture-specific configuration for CPU CI.
#
# This file is sourced by kineto_build_test.sh and pytorch_build_test.sh.
# It defines:
#   - Extra cmake flags for the libkineto build
#   - Environment variables for the PyTorch build
#   - Deselected pytest tests
#

# --- Kineto cmake flags ---
# Disable all GPU backends for CPU-only builds.

# shellcheck disable=SC2034
KINETO_CMAKE_FLAGS=(
  -DKINETO_BACKEND=cpu
)

# --- PyTorch build environment variables ---

export USE_CUDA=0
export USE_CUDNN=0
export USE_NCCL=0
export USE_ROCM=0
export BUILD_TEST=1

# --- PyTorch build caching ---
# This arch's CI runner is an AWS instance, so it can reach PyTorch's shared
# S3 sccache bucket.
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
