#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Architecture-specific configuration for XPU CI.

# --- Kineto cmake flags ---
# Enable XPU (XPUPTI) and disable CUPTI/ROCm backends.
# shellcheck disable=SC2034
KINETO_CMAKE_FLAGS=(
  -DKINETO_BACKEND=xpu
)

# --- PyTorch build environment variables ---
# We're following the pattern established in pytorch/pytorch XPU builds:
#   https://github.com/pytorch/pytorch/blob/39565a7dcf8f93ea22cedeaa20088b24ff6d2634/.ci/manywheel/build_xpu.sh#L20-L28

# We cannot follow through to these files because they only exist on the runners
# shellcheck disable=SC1091
set +u
source /opt/intel/oneapi/compiler/latest/env/vars.sh
source /opt/intel/oneapi/pti/latest/env/vars.sh
if [ -f /opt/intel/oneapi/umf/latest/env/vars.sh ]; then
  source /opt/intel/oneapi/umf/latest/env/vars.sh
fi
if [ -f /opt/intel/oneapi/tcm/latest/env/vars.sh ]; then
  source /opt/intel/oneapi/tcm/latest/env/vars.sh
fi
source /opt/intel/oneapi/ccl/latest/env/vars.sh
source /opt/intel/oneapi/mpi/latest/env/vars.sh
set -u

export USE_STATIC_MKL=1
export USE_XCCL=1
export USE_XPU=1
export BUILD_TEST=1
export USE_CUDA=0
export USE_CUDNN=0
export USE_NCCL=0
export USE_ROCM=0
export USE_MPI=0

# If we don't set this, the logs get flooded with:
#   Double arithmetic operation is not supported on this platform with FP64
#   conversion emulation mode (poison FP64 kernels is enabled)
# TODO: better explanation
export TORCH_XPU_ARCH_LIST=pvc

# --- PyTorch build caching ---
# This arch's CI runner is not on AWS and cannot reach PyTorch's S3 sccache
# bucket.
# shellcheck disable=SC2034
KINETO_USE_SCCACHE=0

# --- Deselected PyTorch profiler tests ---
# Each entry is a pytest node ID passed as a --deselect argument.
#
# shellcheck disable=SC2034
DESELECTED_TESTS=(
  test/profiler/test_profiler.py::TestExperimentalUtils::test_fuzz_symbolize

  # https://github.com/pytorch/kineto/issues/1429
  # Moved into the device-parametrized TestProfilerDevice by pytorch/pytorch#182434.
  # fork-after-init: re-initializing XPU in a forked subprocess raises
  # "Cannot re-initialize XPU in forked subprocess". Independent of runtime XPU
  # availability and already deselected for CUDA and ROCm.
  test/profiler/test_profiler.py::TestProfilerDeviceCPU::test_forked_process_cpu
)
