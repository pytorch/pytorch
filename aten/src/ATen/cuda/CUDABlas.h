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
#if !defined(__HIP_PLATFORM_HCC__) || (defined(__HIP_PLATFORM_HCC__) && HIP_VERSION >= 210)
  template <>
  void gemm<c10::complex<double>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<double>));
#endif
#if !defined(__HIP_PLATFORM_HCC__) || (defined(__HIP_PLATFORM_HCC__) && HIP_VERSION >= 210)
  template <>
  void gemm<c10::complex<float>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<float>));
#endif
template <>
void gemm<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half));
#if defined(__HIP_PLATFORM_HCC__) || defined(CUDA_VERSION) && CUDA_VERSION >= 11000
template <>
void gemm<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16));
#endif

#define CUDABLAS_BGEMM_ARGTYPES(Dtype)                                       \
  char transa, char transb, int64_t m, int64_t n, int64_t k, Dtype alpha,   \
      const Dtype *a, int64_t lda, int64_t stridea, \
      const Dtype *b, int64_t ldb, int64_t strideb, \
      Dtype beta, Dtype *c, int64_t ldc, int64_t stridec, int64_t num_batches

template <typename Dtype>
inline void bgemm(CUDABLAS_BGEMM_ARGTYPES(Dtype)) {
  AT_ERROR("at::cuda::blas::bgemm: not implemented for ", typeid(Dtype).name());
}

template <>
void bgemm<double>(CUDABLAS_BGEMM_ARGTYPES(double));
template <>
void bgemm<float>(CUDABLAS_BGEMM_ARGTYPES(float));
template <>
void bgemm<c10::complex<double>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<double>));
template <>
void bgemm<c10::complex<float>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<float>));
template <>
void bgemm<at::Half>(CUDABLAS_BGEMM_ARGTYPES(at::Half));
#if defined(__HIP_PLATFORM_HCC__) || defined(CUDA_VERSION) && CUDA_VERSION >= 11000
template <>
void bgemm<at::BFloat16>(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
#endif

#define CUDABLAS_TRSM_ARGTYPES(Dtype)                                  \
  cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, \
      cublasOperation_t trans, cublasDiagType_t diag, int m, int n,    \
      const Dtype *alpha, const Dtype *A, int lda, Dtype *B, int ldb

template <typename Dtype>
inline void trsm(CUDABLAS_TRSM_ARGTYPES(Dtype)) {
  TORCH_INTERNAL_ASSERT(false, "at::cuda::blas::trsm: not implemented for ", typeid(Dtype).name());
}

template <>
void trsm<float>(CUDABLAS_TRSM_ARGTYPES(float));
template <>
void trsm<double>(CUDABLAS_TRSM_ARGTYPES(double));
template <>
void trsm<c10::complex<float>>(CUDABLAS_TRSM_ARGTYPES(c10::complex<float>));
template <>
void trsm<c10::complex<double>>(CUDABLAS_TRSM_ARGTYPES(c10::complex<double>));

#define CUDABLAS_TRSM_BATCHED_ARGTYPES(Dtype)                          \
  cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, \
      cublasOperation_t trans, cublasDiagType_t diag, int m, int n,    \
      const Dtype *alpha, Dtype *A[], int lda, Dtype *B[], int ldb,    \
      int batchCount

template <typename Dtype>
inline void trsmBatched(CUDABLAS_TRSM_BATCHED_ARGTYPES(Dtype)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::blas::trsmBatched: not implemented for ",
      typeid(Dtype).name());
}

template <>
void trsmBatched<float>(CUDABLAS_TRSM_BATCHED_ARGTYPES(float));
template <>
void trsmBatched<double>(CUDABLAS_TRSM_BATCHED_ARGTYPES(double));
template <>
void trsmBatched<c10::complex<float>>(CUDABLAS_TRSM_BATCHED_ARGTYPES(c10::complex<float>));
template <>
void trsmBatched<c10::complex<double>>(CUDABLAS_TRSM_BATCHED_ARGTYPES(c10::complex<double>));

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
#if !defined(__HIP_PLATFORM_HCC__) || (defined(__HIP_PLATFORM_HCC__) && HIP_VERSION >= 210)
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
void dot<at::BFloat16>(CUDABLAS_DOT_ARGTYPES(at::BFloat16));
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

// This guards blocks use of geqrfBatched, getrfBatched, getriBatched on platforms other than cuda
#ifdef CUDART_VERSION

#define CUDABLAS_GEQRF_BATCHED_ARGTYPES(Dtype)                   \
  cublasHandle_t handle, int m, int n, Dtype **A_array, int lda, \
      Dtype **tau_array, int *info, int batchsize

template <class Dtype>
void geqrfBatched(CUDABLAS_GEQRF_BATCHED_ARGTYPES(Dtype)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::blas::geqrfBatched: not implemented for ",
      typeid(Dtype).name());
}
template <>
void geqrfBatched<float>(CUDABLAS_GEQRF_BATCHED_ARGTYPES(float));
template <>
void geqrfBatched<double>(CUDABLAS_GEQRF_BATCHED_ARGTYPES(double));
template <>
void geqrfBatched<c10::complex<double>>(
    CUDABLAS_GEQRF_BATCHED_ARGTYPES(c10::complex<double>));
template <>
void geqrfBatched<c10::complex<float>>(
    CUDABLAS_GEQRF_BATCHED_ARGTYPES(c10::complex<float>));

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
template<>
void getrfBatched<c10::complex<double>>(CUDABLAS_GETRF_ARGTYPES(c10::complex<double>));
template<>
void getrfBatched<c10::complex<float>>(CUDABLAS_GETRF_ARGTYPES(c10::complex<float>));


#define CUDABLAS_GETRI_ARGTYPES(Dtype)  \
  int n, Dtype** dA_array, int ldda, int* ipiv_array, Dtype** dC_array, int lddc, int* info_array, int batchsize

template<class Dtype>
void getriBatched(CUDABLAS_GETRI_ARGTYPES(Dtype)) {
  TORCH_CHECK(false, "at::cuda::blas::getriBatched: not implemented for ", typeid(Dtype).name());
}
template<>
void getriBatched<float>(CUDABLAS_GETRI_ARGTYPES(float));
template<>
void getriBatched<double>(CUDABLAS_GETRI_ARGTYPES(double));
template<>
void getriBatched<c10::complex<double>>(CUDABLAS_GETRI_ARGTYPES(c10::complex<double>));
template<>
void getriBatched<c10::complex<float>>(CUDABLAS_GETRI_ARGTYPES(c10::complex<float>));

#define CUDABLAS_GELS_BATCHED_ARGTYPES(Dtype)  \
  cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, Dtype** dA_array, int ldda, Dtype** dC_array, int lddc, int* info, int *devInfoArray, int batchSize

template <class Dtype>
void gelsBatched(CUDABLAS_GELS_BATCHED_ARGTYPES(Dtype)) {
  TORCH_INTERNAL_ASSERT(false, "at::cuda::blas::gelsBatched: not implemented for ", typeid(Dtype).name());
}

template<>
void gelsBatched<double>(CUDABLAS_GELS_BATCHED_ARGTYPES(double));
template<>
void gelsBatched<float>(CUDABLAS_GELS_BATCHED_ARGTYPES(float));
template<>
void gelsBatched<c10::complex<double>>(CUDABLAS_GELS_BATCHED_ARGTYPES(c10::complex<double>));
template<>
void gelsBatched<c10::complex<float>>(CUDABLAS_GELS_BATCHED_ARGTYPES(c10::complex<float>));

#endif // CUDART_VERSION

} // namespace blas
} // namespace cuda
} // namespace at
