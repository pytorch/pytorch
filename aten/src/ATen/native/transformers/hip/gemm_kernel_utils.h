/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This file is a trimmed version of cuda/mem_eff_attention/gemm_kernel_utils.h
#pragma once

#define CHECK_NOSPARSE_CONTIGUOUS_CUDA(TENSOR)                            \
  TORCH_CHECK(TENSOR.is_cuda(), #TENSOR " must be a CUDA tensor");     \
  TORCH_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor"); \
  TORCH_CHECK(TENSOR.is_contiguous());

#define CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(TENSOR)                        \
  TORCH_CHECK(TENSOR.is_cuda(), #TENSOR " must be a CUDA tensor");     \
  TORCH_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor"); \
  TORCH_CHECK(                                                         \
      TENSOR.stride(-1) == 1, #TENSOR ": last dimension must be contiguous");

#define CHECK_ALIGNED_PTR(PTR, ALIGNMENT) \
  TORCH_CHECK(                         \
      uint64_t(PTR) % ALIGNMENT == 0, #PTR " is not correctly aligned")

#define ASSIGN_CHECK_OVERFLOW(A, B)                                    \
  {                                                                    \
    A = B;                                                             \
    TORCH_CHECK(                                                    \
        B < std::numeric_limits<decltype(A)>::max(), #B " overflows"); \
  }
