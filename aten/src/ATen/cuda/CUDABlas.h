#pragma once
/*
  Provides a subset of CUDA BLAS functions as templates:

    gemm<Dtype>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
  ldc)

    gemv<Dtype>(transa, m, n, alpha, a, lda, x, incx, beta, y, incy)

  where Dtype is double, float, at::Half or at::BFloat16(ROCm). The functions are
  available in at::cuda::blas namespace.
 */

#include <ATen/cuda/CUDAContext.h>

namespace at {
namespace cuda {
namespace blas {

/* LEVEL 3 BLAS FUNCTIONS */

#define CUDABLAS_GEMM_ARGTYPES(Dtype)                                      \
      char transa, char transb, int64_t m, int64_t n,                      \
      int64_t k, Dtype alpha, const Dtype *a, int64_t lda, const Dtype *b, \
      int64_t ldb, Dtype beta, Dtype *c, int64_t ldc

template <typename Dtype>
inline void gemm(CUDABLAS_GEMM_ARGTYPES(Dtype)) {
  AT_ERROR("at::cuda::blas::gemm: not implemented for ", typeid(Dtype).name());
}

template <>
void gemm<double>(CUDABLAS_GEMM_ARGTYPES(double));
template <>
void gemm<float>(CUDABLAS_GEMM_ARGTYPES(float));
template <>
void gemm<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half));
#ifdef __HIP_PLATFORM_HCC__
template <>
void gemm<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16));
#endif

/* LEVEL 2 BLAS FUNCTIONS */

#define CUDABLAS_GEMV_ARGTYPES(Dtype)                                        \
      char trans, int64_t m, int64_t n, Dtype alpha,                         \
      const Dtype *a, int64_t lda, const Dtype *x, int64_t incx, Dtype beta, \
      Dtype *y, int64_t incy

template <typename Dtype>
inline void gemv(CUDABLAS_GEMV_ARGTYPES(Dtype)) {
  AT_ERROR("at::cuda::blas::gemv: not implemented for ", typeid(Dtype).name());
}

template <>
void gemv<double>(CUDABLAS_GEMV_ARGTYPES(double));
template <>
void gemv<float>(CUDABLAS_GEMV_ARGTYPES(float));
template <>
void gemv<at::Half>(CUDABLAS_GEMV_ARGTYPES(at::Half));
#ifdef __HIP_PLATFORM_HCC__
template <>
void gemv<at::BFloat16>(CUDABLAS_GEMV_ARGTYPES(at::BFloat16));
#endif

} // namespace blas
} // namespace cuda
} // namespace at
