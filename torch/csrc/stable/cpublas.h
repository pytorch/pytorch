#pragma once

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/macros.h>
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

namespace detail {

inline void brgemm(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t ld_a,
    int64_t ld_b,
    int64_t ld_c,
    bool add_C,
    torch::headeronly::ScalarType dtype_a,
    torch::headeronly::ScalarType dtype_b,
    torch::headeronly::ScalarType dtype_c,
    const void* A,
    const void* B,
    void* C,
    bool is_vnni) {
  STABLE_TORCH_ERROR_CODE_CHECK(torch_cpublas_brgemm(
      M,
      N,
      K,
      ld_a,
      ld_b,
      ld_c,
      static_cast<int32_t>(dtype_a),
      static_cast<int32_t>(dtype_b),
      static_cast<int32_t>(dtype_c),
      static_cast<int32_t>(add_C),
      A,
      B,
      C,
      static_cast<int32_t>(is_vnni)));
}

} // namespace detail

/// Stable ABI wrapper for at::native::cpublas::brgemm.
///
/// Computes C = A @ B, or C += A @ B when add_C is true.
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
  detail::brgemm(
      M,
      N,
      K,
      ld_a,
      ld_b,
      ld_c,
      add_C,
      torch::headeronly::ScalarType::Half,
      torch::headeronly::ScalarType::Half,
      torch::headeronly::ScalarType::Float,
      A,
      B,
      C,
      is_vnni);
}

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
  detail::brgemm(
      M,
      N,
      K,
      ld_a,
      ld_b,
      ld_c,
      add_C,
      torch::headeronly::ScalarType::BFloat16,
      torch::headeronly::ScalarType::BFloat16,
      torch::headeronly::ScalarType::Float,
      A,
      B,
      C,
      is_vnni);
}

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
  detail::brgemm(
      M,
      N,
      K,
      ld_a,
      ld_b,
      ld_c,
      add_C,
      torch::headeronly::ScalarType::Float,
      torch::headeronly::ScalarType::Float,
      torch::headeronly::ScalarType::Float,
      A,
      B,
      C,
      is_vnni);
}

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
  detail::brgemm(
      M,
      N,
      K,
      ld_a,
      ld_b,
      ld_c,
      add_C,
      torch::headeronly::ScalarType::Byte,
      torch::headeronly::ScalarType::Byte,
      torch::headeronly::ScalarType::Int,
      A,
      B,
      C,
      is_vnni);
}

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
  detail::brgemm(
      M,
      N,
      K,
      ld_a,
      ld_b,
      ld_c,
      add_C,
      torch::headeronly::ScalarType::Byte,
      torch::headeronly::ScalarType::Char,
      torch::headeronly::ScalarType::Int,
      A,
      B,
      C,
      is_vnni);
}

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
  detail::brgemm(
      M,
      N,
      K,
      ld_a,
      ld_b,
      ld_c,
      add_C,
      torch::headeronly::ScalarType::Char,
      torch::headeronly::ScalarType::Char,
      torch::headeronly::ScalarType::Int,
      A,
      B,
      C,
      is_vnni);
}

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
  detail::brgemm(
      M,
      N,
      K,
      ld_a,
      ld_b,
      ld_c,
      add_C,
      torch::headeronly::ScalarType::Float8_e4m3fn,
      torch::headeronly::ScalarType::Float8_e4m3fn,
      torch::headeronly::ScalarType::Float,
      A,
      B,
      C,
      is_vnni);
}

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
  detail::brgemm(
      M,
      N,
      K,
      ld_a,
      ld_b,
      ld_c,
      add_C,
      torch::headeronly::ScalarType::Float8_e5m2,
      torch::headeronly::ScalarType::Float8_e5m2,
      torch::headeronly::ScalarType::Float,
      A,
      B,
      C,
      is_vnni);
}

/// Stable ABI wrapper for at::native::cpublas::brgemm_release.
///
/// Minimum compatible version: PyTorch 2.16.
/// Build time minimum version: PyTorch 2.16.
inline void brgemm_release(bool is_vnni = true) {
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_cpublas_brgemm_release(static_cast<int32_t>(is_vnni)));
}

/// Stable C++ interface for the CPU GEMM dispatch stub.
///
/// DEFAULT calls the baseline kernel; operator() uses device dispatch.
/// Both call forms accept double alpha and beta only.
///
/// Minimum compatible version: PyTorch 2.16.
/// Build time minimum version: PyTorch 2.16.
struct GemmNoDowncastStub {
  /// Function pointer to the default CPU kernel's stable shim wrapper.
  static constexpr auto DEFAULT = +[](torch::headeronly::ScalarType dtype,
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
                                      int64_t ldc) {
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_cpublas_gemm_no_downcast_default_alpha_beta_double(
            static_cast<int32_t>(dtype),
            static_cast<int32_t>(transa),
            static_cast<int32_t>(transb),
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
  };

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
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_cpublas_gemm_no_downcast_alpha_beta_double(
            static_cast<int32_t>(device_type),
            static_cast<int32_t>(dtype),
            static_cast<int32_t>(transa),
            static_cast<int32_t>(transb),
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

inline constexpr GemmNoDowncastStub gemm_no_downcast_stub{};

/// Stable ABI wrapper for at::native::cpublas::could_pack.
///
/// Minimum compatible version: PyTorch 2.16.
/// Build time minimum version: PyTorch 2.16.
inline bool could_pack(torch::headeronly::ScalarType dtype) {
  bool ret = false;
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_cpublas_could_pack(static_cast<int32_t>(dtype), &ret));
  return ret;
}

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

HIDDEN_NAMESPACE_END(torch, stable, cpublas)
