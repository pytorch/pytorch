#pragma once

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/version.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/core/TransposeType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Float8_e4m3fn.h>
#include <torch/headeronly/util/Float8_e5m2.h>
#include <torch/headeronly/util/Half.h>
#include <torch/headeronly/util/shim_utils.h>

#include <cstdint>

HIDDEN_NAMESPACE_BEGIN(torch, stable, cpublas)

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

// Match CPUBlas.h's compile-time BRGEMM API availability macros. These signal
// that overloads are declared, not that the CPU supports their kernel paths.
#define CPUBLAS_BRGEMM_F16F16F32 // half * half -> float
#define CPUBLAS_BRGEMM_BF16BF16F32 // bfloat16 * bfloat16 -> float
#define CPUBLAS_BRGEMM_F32F32F32 // float * float -> float
#define CPUBLAS_BRGEMM_U8U8I32 // unsigned char * unsigned char -> int32
#define CPUBLAS_BRGEMM_U8I8I32 // unsigned char * signed char -> int32
#define CPUBLAS_BRGEMM_I8I8I32 // signed char * signed char -> int32
#define CPUBLAS_BRGEMM_F8F8F32 // float8 * float8 -> float (e4m3 & e5m2)

namespace detail {

template <typename AType, typename BType, typename CType>
inline void brgemm(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t ld_a,
    int64_t ld_b,
    int64_t ld_c,
    bool add_C,
    const AType* A,
    const BType* B,
    CType* C,
    bool is_vnni) {
  STABLE_TORCH_ERROR_CODE_CHECK(torch_cpublas_brgemm(
      M,
      N,
      K,
      ld_a,
      ld_b,
      ld_c,
      torch::stable::detail::to<int32_t>(torch::stable::detail::from(
          torch::headeronly::CppTypeToScalarType<AType>::value)),
      torch::stable::detail::to<int32_t>(torch::stable::detail::from(
          torch::headeronly::CppTypeToScalarType<BType>::value)),
      torch::stable::detail::to<int32_t>(torch::stable::detail::from(
          torch::headeronly::CppTypeToScalarType<CType>::value)),
      add_C,
      A,
      B,
      C,
      is_vnni));
}

struct GemmNoDowncastStub {
  void operator()(
      torch::headeronly::DeviceType device_type,
      torch::headeronly::ScalarType dtype,
      torch::headeronly::TransposeType transa,
      torch::headeronly::TransposeType transb,
      int64_t m,
      int64_t n,
      int64_t k,
      double alpha,
      const void* a,
      int64_t lda,
      const void* b,
      int64_t ldb,
      double beta,
      void* c,
      int64_t ldc) const {
    STABLE_TORCH_ERROR_CODE_CHECK(torch_cpublas_gemm_no_downcast_float64(
        torch::stable::detail::to<int32_t>(
            torch::stable::detail::from(device_type)),
        torch::stable::detail::to<int32_t>(torch::stable::detail::from(dtype)),
        torch::stable::detail::to<int32_t>(torch::stable::detail::from(transa)),
        torch::stable::detail::to<int32_t>(torch::stable::detail::from(transb)),
        m,
        n,
        k,
        alpha,
        a,
        lda,
        b,
        ldb,
        beta,
        c,
        ldc));
  }
};

} // namespace detail

/// Stable ABI wrapper for at::native::cpublas::brgemm.
///
/// Computes C = A @ B, or C += A @ B when add_C is true, with Half inputs and
/// float output.
///
/// Minimum compatible version: PyTorch 2.16.
/// Build time minimum version: PyTorch 2.16.
inline void brgemm(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t ld_a,
    int64_t ld_b,
    int64_t ld_c,
    const bool add_C,
    const torch::headeronly::Half* A,
    const torch::headeronly::Half* B,
    float* C,
    bool is_vnni = true) {
  detail::brgemm(M, N, K, ld_a, ld_b, ld_c, add_C, A, B, C, is_vnni);
}

/// BRGEMM with BFloat16 inputs and float output.
///
/// Minimum compatible version: PyTorch 2.16.
/// Build time minimum version: PyTorch 2.16.
inline void brgemm(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t ld_a,
    int64_t ld_b,
    int64_t ld_c,
    const bool add_C,
    const torch::headeronly::BFloat16* A,
    const torch::headeronly::BFloat16* B,
    float* C,
    bool is_vnni = true) {
  detail::brgemm(M, N, K, ld_a, ld_b, ld_c, add_C, A, B, C, is_vnni);
}

/// BRGEMM with float inputs and output; VNNI defaults to false.
///
/// Minimum compatible version: PyTorch 2.16.
/// Build time minimum version: PyTorch 2.16.
inline void brgemm(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t ld_a,
    int64_t ld_b,
    int64_t ld_c,
    const bool add_C,
    const float* A,
    const float* B,
    float* C,
    bool is_vnni = false) {
  detail::brgemm(M, N, K, ld_a, ld_b, ld_c, add_C, A, B, C, is_vnni);
}

/// BRGEMM with unsigned byte inputs and int32 output.
///
/// Minimum compatible version: PyTorch 2.16.
/// Build time minimum version: PyTorch 2.16.
inline void brgemm(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t ld_a,
    int64_t ld_b,
    int64_t ld_c,
    const bool add_C,
    const unsigned char* A,
    const unsigned char* B,
    int32_t* C,
    bool is_vnni = true) {
  detail::brgemm(M, N, K, ld_a, ld_b, ld_c, add_C, A, B, C, is_vnni);
}

/// BRGEMM with unsigned A and signed B byte inputs and int32 output.
///
/// Minimum compatible version: PyTorch 2.16.
/// Build time minimum version: PyTorch 2.16.
inline void brgemm(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t ld_a,
    int64_t ld_b,
    int64_t ld_c,
    const bool add_C,
    const unsigned char* A,
    const signed char* B,
    int32_t* C,
    bool is_vnni = true) {
  detail::brgemm(M, N, K, ld_a, ld_b, ld_c, add_C, A, B, C, is_vnni);
}

/// BRGEMM with signed byte inputs and int32 output.
///
/// Minimum compatible version: PyTorch 2.16.
/// Build time minimum version: PyTorch 2.16.
inline void brgemm(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t ld_a,
    int64_t ld_b,
    int64_t ld_c,
    const bool add_C,
    const signed char* A,
    const signed char* B,
    int32_t* C,
    bool is_vnni = true) {
  detail::brgemm(M, N, K, ld_a, ld_b, ld_c, add_C, A, B, C, is_vnni);
}

/// BRGEMM with Float8_e4m3fn inputs and float output.
///
/// Minimum compatible version: PyTorch 2.16.
/// Build time minimum version: PyTorch 2.16.
inline void brgemm(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t ld_a,
    int64_t ld_b,
    int64_t ld_c,
    const bool add_C,
    const torch::headeronly::Float8_e4m3fn* A,
    const torch::headeronly::Float8_e4m3fn* B,
    float* C,
    bool is_vnni = true) {
  detail::brgemm(M, N, K, ld_a, ld_b, ld_c, add_C, A, B, C, is_vnni);
}

/// BRGEMM with Float8_e5m2 inputs and float output.
///
/// Minimum compatible version: PyTorch 2.16.
/// Build time minimum version: PyTorch 2.16.
inline void brgemm(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t ld_a,
    int64_t ld_b,
    int64_t ld_c,
    const bool add_C,
    const torch::headeronly::Float8_e5m2* A,
    const torch::headeronly::Float8_e5m2* B,
    float* C,
    bool is_vnni = true) {
  detail::brgemm(M, N, K, ld_a, ld_b, ld_c, add_C, A, B, C, is_vnni);
}

/// Stable ABI wrapper for at::native::cpublas::brgemm_release.
///
/// Minimum compatible version: PyTorch 2.16.
/// Build time minimum version: PyTorch 2.16.
inline void brgemm_release(bool is_vnni = true) {
  STABLE_TORCH_ERROR_CODE_CHECK(torch_cpublas_brgemm_release(is_vnni));
}

/// Stable C++ interface for the CPU GEMM dispatch stub.
///
/// Matrices use column-major BLAS layout; lda, ldb, and ldc are leading
/// dimensions in elements. For Half, BFloat16, and Float8 inputs, c must point
/// to a float (opmath) buffer sized for the output matrix and its ldc stride.
/// Passing an input-dtype buffer can write past its end.
///
/// Unlike the native stub's Scalar arguments, alpha and beta are doubles.
/// Complex scalars cannot be passed, and integers above 2^53 may lose
/// precision when converted to double.
///
/// Only DeviceType::CPU has a registered kernel; other device types fail with
/// a missing-kernel internal assertion. This operator dispatches CPU calls to
/// an available CPU capability (such as AVX2, AVX512, or SVE).
///
/// Minimum compatible version: PyTorch 2.16.
/// Build time minimum version: PyTorch 2.16.
inline constexpr detail::GemmNoDowncastStub gemm_no_downcast_stub{};

/// Stable ABI wrapper for at::native::cpublas::could_pack.
///
/// Minimum compatible version: PyTorch 2.16.
/// Build time minimum version: PyTorch 2.16.
inline bool could_pack(torch::headeronly::ScalarType dtype) {
  bool ret = false;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_cpublas_could_pack(
      torch::stable::detail::to<int32_t>(torch::stable::detail::from(dtype)),
      &ret));
  return ret;
}

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

HIDDEN_NAMESPACE_END(torch, stable, cpublas)
