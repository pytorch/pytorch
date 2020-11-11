#pragma once
/*
  Provides a subset of CUDA BLAS functions as templates:

    gemm<Dtype>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
  ldc)

    gemv<Dtype>(transa, m, n, alpha, a, lda, x, incx, beta, y, incy)

    dot<Dtype>(n, x, incx, y, incy, result)

  where Dtype is double, float, at::Half or at::BFloat16 (ROCm, NOT for dot).
  The functions are available in at::cuda::blas namespace.
 */

#include <ATen/cuda/CUDAContext.h>

namespace at {
namespace cuda {
namespace blas {

// RAII guard that sets the CuBLAS pointer mode and restores it to
// its previous value when the guard is destroyed
class PointerModeGuard {
public:
  PointerModeGuard(cublasHandle_t handle, cublasPointerMode_t mode) :
      handle(handle) {
    TORCH_CUDABLAS_CHECK(cublasGetPointerMode(handle, &previous_mode));
    TORCH_CUDABLAS_CHECK(cublasSetPointerMode(handle, mode));
  }

  ~PointerModeGuard() {
    cublasSetPointerMode(handle, previous_mode);
  }

private:
  cublasHandle_t handle;
  cublasPointerMode_t previous_mode;
};

/* LEVEL 3 BLAS FUNCTIONS */

#define CUDABLAS_GEMM_ARGTYPES(Dtype)                                       \
  char transa, char transb, int64_t m, int64_t n, int64_t k, Dtype alpha,   \
      const Dtype *a, int64_t lda, const Dtype *b, int64_t ldb, Dtype beta, \
      Dtype *c, int64_t ldc

template <typename Dtype>
inline void gemm(CUDABLAS_GEMM_ARGTYPES(Dtype)) {
  AT_ERROR("at::cuda::blas::gemm: not implemented for ", typeid(Dtype).name());
}

template <>
void gemm<double>(CUDABLAS_GEMM_ARGTYPES(double));
template <>
void gemm<float>(CUDABLAS_GEMM_ARGTYPES(float));
#ifndef __HIP_PLATFORM_HCC__
  template <>
  void gemm<c10::complex<double>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<double>));
#endif
#ifndef __HIP_PLATFORM_HCC__
  template <>
  void gemm<c10::complex<float>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<float>));
#endif
template <>
void gemm<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half));
#if defined(__HIP_PLATFORM_HCC__) || defined(CUDA_VERSION) && CUDA_VERSION >= 11000
template <>
void gemm<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16));
#endif

/* LEVEL 2 BLAS FUNCTIONS */

#define CUDABLAS_GEMV_ARGTYPES(Dtype)                                         \
  char trans, int64_t m, int64_t n, Dtype alpha, const Dtype *a, int64_t lda, \
      const Dtype *x, int64_t incx, Dtype beta, Dtype *y, int64_t incy

template <typename Dtype>
inline void gemv(CUDABLAS_GEMV_ARGTYPES(Dtype)) {
  AT_ERROR("at::cuda::blas::gemv: not implemented for ", typeid(Dtype).name());
}

template <>
void gemv<double>(CUDABLAS_GEMV_ARGTYPES(double));
template <>
void gemv<float>(CUDABLAS_GEMV_ARGTYPES(float));
#ifndef __HIP_PLATFORM_HCC__
template <>
void gemv<c10::complex<double>>(CUDABLAS_GEMV_ARGTYPES(c10::complex<double>));
template <>
void gemv<c10::complex<float>>(CUDABLAS_GEMV_ARGTYPES(c10::complex<float>));
#endif
template <>
void gemv<at::Half>(CUDABLAS_GEMV_ARGTYPES(at::Half));
#if defined(__HIP_PLATFORM_HCC__) || defined(CUDA_VERSION) && CUDA_VERSION >= 11000
template <>
void gemv<at::BFloat16>(CUDABLAS_GEMV_ARGTYPES(at::BFloat16));
#endif

template <typename Dtype>
void ger(
    int64_t m,
    int64_t n,
    Dtype alpha,
    Dtype* x,
    int64_t incx,
    Dtype* y,
    int64_t incy,
    Dtype* a,
    int64_t lda);

/* LEVEL 1 BLAS FUNCTIONS */

#define CUDABLAS_DOT_ARGTYPES(Dtype)                                      \
  cublasHandle_t handle, int n, const Dtype *x, int incx, const Dtype *y, \
      int incy, Dtype *result

template <typename Dtype>
inline void dot(CUDABLAS_DOT_ARGTYPES(Dtype)) {
  AT_ERROR("at::cuda::blas::dot: not implemented for ", typeid(Dtype).name());
}

template <>
void dot<double>(CUDABLAS_DOT_ARGTYPES(double));
template <>
void dot<float>(CUDABLAS_DOT_ARGTYPES(float));
template <>
void dot<at::Half>(CUDABLAS_DOT_ARGTYPES(at::Half));
template <>
void dot<c10::complex<double>>(CUDABLAS_DOT_ARGTYPES(c10::complex<double>));
template <>
void dot<c10::complex<float>>(CUDABLAS_DOT_ARGTYPES(c10::complex<float>));

template <typename Dtype>
inline void vdot(CUDABLAS_DOT_ARGTYPES(Dtype)) {
  AT_ERROR("at::cuda::blas::vdot: not implemented for ", typeid(Dtype).name());
}

template <>
void vdot<c10::complex<float>>(CUDABLAS_DOT_ARGTYPES(c10::complex<float>));
template <>
void vdot<c10::complex<double>>(CUDABLAS_DOT_ARGTYPES(c10::complex<double>));

// This guards blocks use of getrfBatched and getriBatched on platforms other than cuda
#ifdef CUDART_VERSION

#define CUDABLAS_GETRF_ARGTYPES(Dtype)  \
  int n, Dtype** dA_array, int ldda, int* ipiv_array, int* info_array, int batchsize

template<class Dtype>
void getrfBatched(CUDABLAS_GETRF_ARGTYPES(Dtype)) {
  TORCH_CHECK(false, "at::cuda::blas::getrfBatched: not implemented for ", typeid(Dtype).name());
}
template<>
void getrfBatched<float>(CUDABLAS_GETRF_ARGTYPES(float));
template<>
void getrfBatched<double>(CUDABLAS_GETRF_ARGTYPES(double));


#define CUDABLAS_GETRI_ARGTYPES(Dtype)  \
  int n, Dtype** dA_array, int ldda, int* ipiv_array, int* info_array, int batchsize, Dtype** dC_array

template<class Dtype>
void getriBatched(CUDABLAS_GETRI_ARGTYPES(Dtype)) {
  TORCH_CHECK(false, "at::cuda::blas::getriBatched: not implemented for ", typeid(Dtype).name());
}
template<>
void getriBatched<float>(CUDABLAS_GETRI_ARGTYPES(float));
template<>
void getriBatched<double>(CUDABLAS_GETRI_ARGTYPES(double));

#endif // CUDART_VERSION

} // namespace blas
} // namespace cuda
} // namespace at
