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
#include <ATen/OpMathType.h>

namespace at::cuda::blas {

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
  cublasPointerMode_t previous_mode{};
};

/* LEVEL 3 BLAS FUNCTIONS */

#define CUDABLAS_GEMM_ARGTYPES(Dtype) CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(Dtype, Dtype)

#define CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)                                  \
  char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<Dtype> alpha,  \
      const Dtype *a, int64_t lda, const Dtype *b, int64_t ldb, at::opmath_type<Dtype> beta,\
      C_Dtype *c, int64_t ldc

#define CUDABLAS_GEMM_ARGS(Dtype) transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc

#define CUDABLAS_GEMM_DTYPE_IS_FLOAT_TYPE_AND_C_DTYPE_IS_FLOAT \
    ((std::is_same<Dtype, at::Half>::value || std::is_same<Dtype, at::BFloat16>::value) && std::is_same<C_Dtype, float>::value)

template <typename Dtype, typename C_Dtype = Dtype, typename std::enable_if<!CUDABLAS_GEMM_DTYPE_IS_FLOAT_TYPE_AND_C_DTYPE_IS_FLOAT, Dtype>::type* = nullptr>
inline void gemm(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)) {
  static_assert(false&&sizeof(Dtype),"at::cuda::blas::gemm: not implemented");
}

template <typename Dtype, typename C_Dtype, typename std::enable_if<CUDABLAS_GEMM_DTYPE_IS_FLOAT_TYPE_AND_C_DTYPE_IS_FLOAT, Dtype>::type* = nullptr>
void gemm(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype));

template <>
void gemm<double>(CUDABLAS_GEMM_ARGTYPES(double));
template <>
void gemm<float>(CUDABLAS_GEMM_ARGTYPES(float));
template <>
void gemm<c10::complex<double>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<double>));
template <>
void gemm<c10::complex<float>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<float>));
template <>
void gemm<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half));
template <>
void gemm<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16));
template<>
void gemm<at::Half, float>(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::Half, float));
template<>
void gemm<at::BFloat16, float>(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float));

template <typename Dtype, typename C_Dtype = Dtype>
inline void gemm_internal(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)) {
  static_assert(false&&sizeof(Dtype),"at::cuda::blas::gemm_internal: not implemented");
}

template <>
void gemm_internal<double>(CUDABLAS_GEMM_ARGTYPES(double));
template <>
void gemm_internal<float>(CUDABLAS_GEMM_ARGTYPES(float));
template <>
void gemm_internal<c10::complex<double>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<double>));
template <>
void gemm_internal<c10::complex<float>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<float>));
template <>
void gemm_internal<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half));
template <>
void gemm_internal<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16));
template<>
void gemm_internal<at::Half, float>(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::Half, float));
template<>
void gemm_internal<at::BFloat16, float>(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float));

enum GEMMAndBiasActivationEpilogue {
  None,
  RELU,
  GELU,
};

// NOTE: GELU activation is not supported prior to CUDA 11.4 and will
// do nothing if passed in that case.
template <typename Dtype, typename C_Dtype = Dtype>
bool gemm_and_bias(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    at::opmath_type<Dtype> alpha_val,
    const Dtype* mat1_ptr,
    int64_t mat1_ld,
    const Dtype* mat2_ptr,
    int64_t mat2_ld,
    const Dtype* bias,
    C_Dtype* result_ptr,
    int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation = GEMMAndBiasActivationEpilogue::None);

void int8_gemm(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    const int8_t* mat1_ptr,
    int64_t mat1_ld,
    const int8_t* mat2_ptr,
    int64_t mat2_ld,
    int32_t* result_ptr,
    int64_t result_ld);

void scaled_gemm(
    char transa,
    char transb,
    int64_t m,
    int64_t n,
    int64_t k,
    const void* mat1_ptr,
    const void* mat1_scale_ptr,
    int64_t mat1_ld,
    ScalarType mat1_dtype,
    ScalarType mat1_scale_dtype,
    const void* mat2_ptr,
    const void* mat2_scale_ptr,
    int64_t mat2_ld,
    ScalarType mat2_dtype,
    ScalarType mat2_scale_dtype,
    const void* bias_ptr,
    ScalarType bias_dtype,
    void* result_ptr,
    const void* result_scale_ptr,
    int64_t result_ld,
    ScalarType result_dtype,
    bool use_fast_accum,
    bool use_rowwise);

#define CUDABLAS_BGEMM_ARGTYPES(Dtype)  CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(Dtype, Dtype)

#define CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)                                   \
  char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<Dtype> alpha,    \
      const Dtype *a, int64_t lda, int64_t stridea,                                           \
      const Dtype *b, int64_t ldb, int64_t strideb,                                           \
      at::opmath_type<Dtype> beta, C_Dtype *c, int64_t ldc, int64_t stridec, int64_t num_batches

#define CUDABLAS_BGEMM_ARGS(Dtype) \
  transa, transb, m, n, k, alpha, a, lda, stridea, b, ldb, strideb, beta, c, ldc, stridec, num_batches

template <typename Dtype, typename C_Dtype = Dtype, typename std::enable_if<!CUDABLAS_GEMM_DTYPE_IS_FLOAT_TYPE_AND_C_DTYPE_IS_FLOAT, Dtype>::type* = nullptr>
inline void bgemm(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)) {
  static_assert(false&&sizeof(Dtype),"at::cuda::blas::bgemm: not implemented");
}

template <typename Dtype, typename C_Dtype, typename std::enable_if<CUDABLAS_GEMM_DTYPE_IS_FLOAT_TYPE_AND_C_DTYPE_IS_FLOAT, Dtype>::type* = nullptr>
void bgemm(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype));

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
template <>
void bgemm<at::BFloat16>(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
template<>
void bgemm<at::Half, float>(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::Half, float));
template<>
void bgemm<at::BFloat16, float>(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float));

template <typename Dtype, typename C_Dtype = Dtype>
inline void bgemm_internal(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)) {
  static_assert(false&&sizeof(Dtype),"at::cuda::blas::bgemm_internal: not implemented");
}

template <>
void bgemm_internal<double>(CUDABLAS_BGEMM_ARGTYPES(double));
template <>
void bgemm_internal<float>(CUDABLAS_BGEMM_ARGTYPES(float));
template <>
void bgemm_internal<c10::complex<double>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<double>));
template <>
void bgemm_internal<c10::complex<float>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<float>));
template <>
void bgemm_internal<at::Half>(CUDABLAS_BGEMM_ARGTYPES(at::Half));
template <>
void bgemm_internal<at::BFloat16>(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));
template<>
void bgemm_internal<at::Half, float>(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::Half, float));
template<>
void bgemm_internal<at::BFloat16, float>(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float));

#define CUDABLAS_TRSM_ARGTYPES(Dtype)                                  \
  cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, \
      cublasOperation_t trans, cublasDiagType_t diag, int m, int n,    \
      const Dtype *alpha, const Dtype *A, int lda, Dtype *B, int ldb

template <typename Dtype>
inline void trsm(CUDABLAS_TRSM_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype), "at::cuda::blas::trsm: not implemented");
}

template <>
TORCH_CUDA_CU_API void trsm<float>(CUDABLAS_TRSM_ARGTYPES(float));
template <>
TORCH_CUDA_CU_API void trsm<double>(CUDABLAS_TRSM_ARGTYPES(double));
template <>
TORCH_CUDA_CU_API void trsm<c10::complex<float>>(CUDABLAS_TRSM_ARGTYPES(c10::complex<float>));
template <>
TORCH_CUDA_CU_API void trsm<c10::complex<double>>(CUDABLAS_TRSM_ARGTYPES(c10::complex<double>));

#define CUDABLAS_TRSM_BATCHED_ARGTYPES(Dtype)                          \
  cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, \
      cublasOperation_t trans, cublasDiagType_t diag, int m, int n,    \
      const Dtype *alpha, Dtype *A[], int lda, Dtype *B[], int ldb,    \
      int batchCount

template <typename Dtype>
inline void trsmBatched(CUDABLAS_TRSM_BATCHED_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype), "at::cuda::blas::trsmBatched: not implemented");
}

template <>
TORCH_CUDA_CU_API void trsmBatched<float>(CUDABLAS_TRSM_BATCHED_ARGTYPES(float));
template <>
TORCH_CUDA_CU_API void trsmBatched<double>(CUDABLAS_TRSM_BATCHED_ARGTYPES(double));
template <>
TORCH_CUDA_CU_API void trsmBatched<c10::complex<float>>(CUDABLAS_TRSM_BATCHED_ARGTYPES(c10::complex<float>));
template <>
TORCH_CUDA_CU_API void trsmBatched<c10::complex<double>>(CUDABLAS_TRSM_BATCHED_ARGTYPES(c10::complex<double>));

/* LEVEL 2 BLAS FUNCTIONS */

#define CUDABLAS_GEMV_ARGTYPES(Dtype)                                         \
  char trans, int64_t m, int64_t n, Dtype alpha, const Dtype *a, int64_t lda, \
      const Dtype *x, int64_t incx, Dtype beta, Dtype *y, int64_t incy

template <typename Dtype>
inline void gemv(CUDABLAS_GEMV_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype), "at::cuda::blas::gemv: not implemented");
}

template <>
void gemv<double>(CUDABLAS_GEMV_ARGTYPES(double));
template <>
void gemv<float>(CUDABLAS_GEMV_ARGTYPES(float));
template <>
void gemv<c10::complex<double>>(CUDABLAS_GEMV_ARGTYPES(c10::complex<double>));
template <>
void gemv<c10::complex<float>>(CUDABLAS_GEMV_ARGTYPES(c10::complex<float>));
template <>
void gemv<at::Half>(CUDABLAS_GEMV_ARGTYPES(at::Half));
template <>
void gemv<at::BFloat16>(CUDABLAS_GEMV_ARGTYPES(at::BFloat16));

/* LEVEL 1 BLAS FUNCTIONS */

#define CUDABLAS_DOT_ARGTYPES(Dtype)                                      \
  cublasHandle_t handle, int n, const Dtype *x, int incx, const Dtype *y, \
      int incy, Dtype *result

template <typename Dtype>
inline void dot(CUDABLAS_DOT_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype),"at::cuda::blas::dot: not implemented");
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
  static_assert(false&&sizeof(Dtype),"at::cuda::blas::vdot: not implemented");
}

template <>
void vdot<c10::complex<float>>(CUDABLAS_DOT_ARGTYPES(c10::complex<float>));
template <>
void vdot<c10::complex<double>>(CUDABLAS_DOT_ARGTYPES(c10::complex<double>));

#define CUDABLAS_GETRS_ARGTYPES(Dtype)  \
  cublasHandle_t handle, cublasOperation_t trans, \
  int n, int nrhs, Dtype** dA_array, int lda, int* ipiv_array, \
  Dtype** dB_array, int ldb, int* info_array, int batchsize

#define CUDABLAS_GEQRF_BATCHED_ARGTYPES(Dtype)                   \
  cublasHandle_t handle, int m, int n, Dtype **A_array, int lda, \
      Dtype **tau_array, int *info, int batchsize

#define CUDABLAS_GETRF_ARGTYPES(Dtype)  \
  int n, Dtype** dA_array, int ldda, int* ipiv_array, int* info_array, int batchsize

#define CUDABLAS_GELS_BATCHED_ARGTYPES(Dtype)  \
  cublasHandle_t handle, cublasOperation_t trans, \
  int m, int n, int nrhs, Dtype** dA_array, int ldda, \
  Dtype** dC_array, int lddc, int* info, int *devInfoArray, int batchSize

// HIP on Windows does not support getrs, geqrf, getrf, gels
#if !(defined(USE_ROCM) && defined(_MSC_VER))

template<class Dtype>
void getrsBatched(CUDABLAS_GETRS_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype),"at::cuda::blas::getrsBatched: not implemented");
}
template<>
TORCH_CUDA_CU_API void getrsBatched<float>(CUDABLAS_GETRS_ARGTYPES(float));
template<>
TORCH_CUDA_CU_API void getrsBatched<double>(CUDABLAS_GETRS_ARGTYPES(double));
template<>
TORCH_CUDA_CU_API void getrsBatched<c10::complex<float>>(CUDABLAS_GETRS_ARGTYPES(c10::complex<float>));
template<>
TORCH_CUDA_CU_API void getrsBatched<c10::complex<double>>(CUDABLAS_GETRS_ARGTYPES(c10::complex<double>));

template <class Dtype>
void geqrfBatched(CUDABLAS_GEQRF_BATCHED_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype), "at::cuda::blas::geqrfBatched: not implemented");
}
template <>
TORCH_CUDA_CU_API void geqrfBatched<float>(CUDABLAS_GEQRF_BATCHED_ARGTYPES(float));
template <>
TORCH_CUDA_CU_API void geqrfBatched<double>(CUDABLAS_GEQRF_BATCHED_ARGTYPES(double));
template <>
TORCH_CUDA_CU_API void geqrfBatched<c10::complex<double>>(
    CUDABLAS_GEQRF_BATCHED_ARGTYPES(c10::complex<double>));
template <>
TORCH_CUDA_CU_API void geqrfBatched<c10::complex<float>>(
    CUDABLAS_GEQRF_BATCHED_ARGTYPES(c10::complex<float>));

template<class Dtype>
void getrfBatched(CUDABLAS_GETRF_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype), "at::cuda::blas::getrfBatched: not implemented");
}
template<>
TORCH_CUDA_CU_API void getrfBatched<float>(CUDABLAS_GETRF_ARGTYPES(float));
template<>
TORCH_CUDA_CU_API void getrfBatched<double>(CUDABLAS_GETRF_ARGTYPES(double));
template<>
TORCH_CUDA_CU_API void getrfBatched<c10::complex<double>>(CUDABLAS_GETRF_ARGTYPES(c10::complex<double>));
template<>
TORCH_CUDA_CU_API void getrfBatched<c10::complex<float>>(CUDABLAS_GETRF_ARGTYPES(c10::complex<float>));

template <class Dtype>
void gelsBatched(CUDABLAS_GELS_BATCHED_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype), "at::cuda::blas::gelsBatched: not implemented");
}
template<>
TORCH_CUDA_CU_API void gelsBatched<double>(CUDABLAS_GELS_BATCHED_ARGTYPES(double));
template<>
TORCH_CUDA_CU_API void gelsBatched<float>(CUDABLAS_GELS_BATCHED_ARGTYPES(float));
template<>
TORCH_CUDA_CU_API void gelsBatched<c10::complex<double>>(CUDABLAS_GELS_BATCHED_ARGTYPES(c10::complex<double>));
template<>
TORCH_CUDA_CU_API void gelsBatched<c10::complex<float>>(CUDABLAS_GELS_BATCHED_ARGTYPES(c10::complex<float>));

#else // !(defined(USE_ROCM) && defined(_MSC_VER))

template<class Dtype>
void getrsBatched(CUDABLAS_GETRS_ARGTYPES(Dtype)) {
  TORCH_CHECK(false, "at::cuda::blas::getrsBatched: not supported for HIP on Windows");
}

template <class Dtype>
void geqrfBatched(CUDABLAS_GEQRF_BATCHED_ARGTYPES(Dtype)) {
  TORCH_CHECK(false, "at::cuda::blas::geqrfBatched: not supported for HIP on Windows");
}

template<class Dtype>
void getrfBatched(CUDABLAS_GETRF_ARGTYPES(Dtype)) {
  TORCH_CHECK(false, "at::cuda::blas::getrfBatched: not supported for HIP on Windows");
}

template <class Dtype>
void gelsBatched(CUDABLAS_GELS_BATCHED_ARGTYPES(Dtype)) {
  TORCH_CHECK(false, "at::cuda::blas::gelsBatched: not supported for HIP on Windows");
}

#endif // !(defined(USE_ROCM) && defined(_MSC_VER))

} // namespace at::cuda::blas
