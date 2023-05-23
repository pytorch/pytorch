/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cutlass/arch/mma.h>

////////////////////////////////////////////////////////////////////////////////
// Some helper functions
////////////////////////////////////////////////////////////////////////////////
#define DISPATCH_TYPES(tensor, func)                                           \
  {                                                                            \
    if (query.scalar_type() == at::ScalarType::Float) {                        \
      using scalar_t = float;                                                  \
      func();                                                                  \
    } else if (query.scalar_type() == at::ScalarType::Half) {                  \
      using scalar_t = cutlass::half_t;                                        \
      func();                                                                  \
    } else if (query.scalar_type() == at::ScalarType::BFloat16) {              \
      using scalar_t = cutlass::bfloat16_t;                                    \
      func();                                                                  \
    } else {                                                                   \
      TORCH_CHECK(false, "Only fp32, half & bf16 supported at the moment"); \
    }                                                                          \
  }

#define DISPATCH_BOOL(BOOL_V, BOOL_NAME, F) \
  {                                         \
    if (BOOL_V) {                           \
      constexpr bool BOOL_NAME = true;      \
      F();                                  \
    } else {                                \
      constexpr bool BOOL_NAME = false;     \
      F();                                  \
    }                                       \
  }
#define DISPATCH_ARCHTAG(CC, func)                                        \
  {                                                                       \
    if (CC >= 80) {                                                       \
      using ArchTag = cutlass::arch::Sm80;                                \
      func();                                                             \
    } else if (CC >= 75) {                                                \
      using ArchTag = cutlass::arch::Sm75;                                \
      func();                                                             \
    } else if (CC >= 70) {                                                \
      using ArchTag = cutlass::arch::Sm70;                                \
      func();                                                             \
    } else if (CC >= 50) {                                                \
      using ArchTag = cutlass::arch::Sm50;                                \
      func();                                                             \
    } else {                                                              \
      TORCH_CHECK(                                                     \
          false,                                                          \
          "Your device is too old. We require compute capability >= 50"); \
    }                                                                     \
  }

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

namespace gemm_kernel_utils {

template <typename integer>
constexpr CUTLASS_HOST_DEVICE integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

template <typename integer>
constexpr CUTLASS_HOST_DEVICE integer align_up(integer n, integer m) {
  return ((n + m - 1) / m) * m;
}

////////////////////////////////////////////////////////////////////////////////
// Determine the type of GEMM we do (TensorCores or not, Shapes ...)
// TODO: Maybe we could rely on Cutlass's DefaultGemm templates
////////////////////////////////////////////////////////////////////////////////

// Fallback to Simt (FMA on cuda cores) if not in a special case below
template <typename ArchTag, typename scalar_t_, typename Enable = void>
struct DefaultGemmType {
  static constexpr int ThreadK = 8;
  static constexpr int WarpK = 8;
  static constexpr int kMinimumAlignment = 1;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  using OpClass = cutlass::arch::OpClassSimt;
  using Operator = cutlass::arch::OpMultiplyAdd;
};

// Specialization for tensorcores with f32
template <typename ArchTag>
struct DefaultGemmType<
    ArchTag,
    float,
    typename cutlass::platform::enable_if<
        ArchTag::kMinComputeCapability >= 80>::type> {
  static constexpr int ThreadK = 32;
  static constexpr int WarpK = 32;
  static constexpr int kMinimumAlignment = 4;
  using OpClass = cutlass::arch::OpClassTensorOp;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Operator = cutlass::arch::OpMultiplyAddFastF32;
};

// Specialization for tensorcores with f16/bf16 - Sm75+
template <typename ArchTag, typename scalar_t>
struct DefaultGemmType<
    ArchTag,
    scalar_t,
    typename cutlass::platform::enable_if<
        ArchTag::kMinComputeCapability >= 75 &&
        cutlass::sizeof_bits<scalar_t>::value == 16>::type> {
  static constexpr int ThreadK = 32;
  static constexpr int WarpK = 32;
  static constexpr int kMinimumAlignment = 4;
  using OpClass = cutlass::arch::OpClassTensorOp;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Operator = cutlass::arch::OpMultiplyAdd;
};

// Specialization for tensorcores with f16 - Volta
template <>
struct DefaultGemmType<cutlass::arch::Sm70, cutlass::half_t, void> {
  static constexpr int ThreadK = 32;
  static constexpr int WarpK = 32;
  static constexpr int kMinimumAlignment = 2;
  using OpClass = cutlass::arch::OpClassTensorOp;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
};

// Enables to do
// `auto x = kCondition ? fa(arg) : fb(arg)`
// when `fa` and `fb` have different types
template <bool kVal, typename TA, typename TB>
struct call_conditional;

template <typename TA, typename TB>
struct call_conditional<true, TA, TB> {
  template <typename Arg>
  static CUTLASS_HOST_DEVICE auto apply(TA ta, TB tb, Arg arg)
      -> decltype(ta(arg)) {
    return ta(arg);
  }
};

template <typename TA, typename TB>
struct call_conditional<false, TA, TB> {
  template <typename Arg>
  static CUTLASS_HOST_DEVICE auto apply(TA ta, TB tb, Arg arg)
      -> decltype(tb(arg)) {
    return tb(arg);
  }
};

////////////////////////////////////////////////////////////////////////////////
// Mark a variable as warp-uniform - enables some compiler optimizations
// The cheapest way to do it is just to broadcast it from lane 0
////////////////////////////////////////////////////////////////////////////////

CUTLASS_DEVICE int32_t warp_uniform(int32_t value) {
  return (int32_t)__shfl_sync(0xffffffff, (unsigned)value, 0);
}

template <typename T>
CUTLASS_DEVICE T* warp_uniform(T* ptr) {
  struct {
    union {
      T* ptr;
      uint32_t asInt[2];
    };
  } p;
  p.ptr = ptr;
  p.asInt[0] = warp_uniform(p.asInt[0]);
  p.asInt[1] = warp_uniform(p.asInt[1]);
  return p.ptr;
}
} // namespace gemm_kernel_utils
