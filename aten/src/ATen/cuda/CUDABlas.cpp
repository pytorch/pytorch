/*
  Provides the implementations of CUDA BLAS function templates.
 */

#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/Exceptions.h>

#define CUDABLAS_POSINT_CHECK(FD, X)         \
  TORCH_CHECK(                               \
      (X > 0 && X <= INT_MAX),               \
      "at::cuda::blas::" #FD " argument " #X \
      " must be positive and less than ",    \
      INT_MAX,                               \
      " but got ",                           \
      X)

#define CUDABLAS_NONNEGINT_CHECK(FD, X)       \
  TORCH_CHECK(                                \
      (X >= 0 && X <= INT_MAX),               \
      "at::cuda::blas::" #FD " argument " #X  \
      " must be non-negative and less than ", \
      INT_MAX,                                \
      " but got ",                            \
      X)

namespace {

static cublasOperation_t _cublasOpFromChar(char op) {
  switch (op) {
    case 'n':
    case 'N':
      return CUBLAS_OP_N;
    case 't':
    case 'T':
      return CUBLAS_OP_T;
    case 'c':
    case 'C':
      return CUBLAS_OP_C;
  }
  AT_ERROR(
      "_cublasOpFromChar input should be 't', 'n' or 'c' but got `", op, "`");
}

static void _cublasAdjustLdLevel2(int64_t m, int64_t n, int64_t* lda) {
  // Note: leading dimensions generally are checked that they are > 0
  // and at least as big the result requires (even if the value won't
  // be used).

  // Q: Why does Level3 check trans but this doesn't?
  // A: In level 2, the sizes (m, n) specify the size of A
  // (independent of trans value). In level 3. the sizes (m, n, k)
  // specify the sizes of op(A), op(B) where op depend on trans
  // values.
  if (n <= 1)
    *lda = std::max<int64_t>(m, 1);
}

static void _cublasAdjustLdLevel3(
    char transa,
    char transb,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t* lda,
    int64_t* ldb,
    int64_t* ldc) {
  bool transa_ = ((transa == 't') || (transa == 'T'));
  bool transb_ = ((transb == 't') || (transb == 'T'));

  // Note: leading dimensions generally are checked that they are > 0
  // and at least as big the result requires (even if the value won't
  // be used).
  if (n <= 1)
    *ldc = std::max<int64_t>(m, 1);

  if (transa_) {
    if (m <= 1)
      *lda = std::max<int64_t>(k, 1);
  } else {
    if (k <= 1)
      *lda = std::max<int64_t>(m, 1);
  }

  if (transb_) {
    if (k <= 1)
      *ldb = std::max<int64_t>(n, 1);
  } else {
    if (n <= 1)
      *ldb = std::max<int64_t>(k, 1);
  }
}
} // anonymous namespace

namespace at {
namespace cuda {
namespace blas {

const char* _cublasGetErrorEnum(cublasStatus_t error) {
  if (error == CUBLAS_STATUS_SUCCESS) {
    return "CUBLAS_STATUS_SUCCESS";
  }
  if (error == CUBLAS_STATUS_NOT_INITIALIZED) {
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  }
  if (error == CUBLAS_STATUS_ALLOC_FAILED) {
    return "CUBLAS_STATUS_ALLOC_FAILED";
  }
  if (error == CUBLAS_STATUS_INVALID_VALUE) {
    return "CUBLAS_STATUS_INVALID_VALUE";
  }
  if (error == CUBLAS_STATUS_ARCH_MISMATCH) {
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  }
  if (error == CUBLAS_STATUS_MAPPING_ERROR) {
    return "CUBLAS_STATUS_MAPPING_ERROR";
  }
  if (error == CUBLAS_STATUS_EXECUTION_FAILED) {
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  }
  if (error == CUBLAS_STATUS_INTERNAL_ERROR) {
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  if (error == CUBLAS_STATUS_NOT_SUPPORTED) {
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  }
#ifdef CUBLAS_STATUS_LICENSE_ERROR
  if (error == CUBLAS_STATUS_LICENSE_ERROR) {
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }
#endif
  return "<unknown>";
}

/* LEVEL 3 BLAS FUNCTIONS */

#define GEMM_CHECK_ARGVALUES(Dtype)           \
  do {                                        \
    CUDABLAS_NONNEGINT_CHECK(gemm<Dtype>, m); \
    CUDABLAS_NONNEGINT_CHECK(gemm<Dtype>, n); \
    CUDABLAS_NONNEGINT_CHECK(gemm<Dtype>, k); \
    CUDABLAS_POSINT_CHECK(gemm<Dtype>, lda);  \
    CUDABLAS_POSINT_CHECK(gemm<Dtype>, ldb);  \
    CUDABLAS_POSINT_CHECK(gemm<Dtype>, ldc);  \
  } while (0)

template <>
void gemm<double>(CUDABLAS_GEMM_ARGTYPES(double)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(double);
  TORCH_CUDABLAS_CHECK(cublasDgemm(
      handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}

template <>
void gemm<float>(CUDABLAS_GEMM_ARGTYPES(float)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(float);
  TORCH_CUDABLAS_CHECK(cublasSgemm(
      handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}

template <>
void gemm<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  float falpha = alpha;
  float fbeta = beta;
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(at::Half);
#ifdef __HIP_PLATFORM_HCC__
  TORCH_CUDABLAS_CHECK(rocblas_gemm_ex(
      handle,
      opa,
      opb,
      m,
      n,
      k,
      &falpha,
      a,
      rocblas_datatype_f16_r,
      lda,
      b,
      rocblas_datatype_f16_r,
      ldb,
      &fbeta,
      c,
      rocblas_datatype_f16_r,
      ldc,
      c,
      rocblas_datatype_f16_r,
      ldc,
      rocblas_datatype_f32_r,
      rocblas_gemm_algo_standard,
      0,
      0));
#else
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major >= 5) {
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    TORCH_CUDABLAS_CHECK(cublasGemmEx(
        handle,
        opa,
        opb,
        m,
        n,
        k,
        &falpha,
        a,
        CUDA_R_16F,
        lda,
        b,
        CUDA_R_16F,
        ldb,
        &fbeta,
        c,
        CUDA_R_16F,
        ldc,
        CUDA_R_32F,
        CUBLAS_GEMM_DFALT_TENSOR_OP));
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  } else {
    TORCH_CUDABLAS_CHECK(cublasSgemmEx(
        handle,
        opa,
        opb,
        m,
        n,
        k,
        &falpha,
        a,
        CUDA_R_16F,
        lda,
        b,
        CUDA_R_16F,
        ldb,
        &fbeta,
        c,
        CUDA_R_16F,
        ldc));
  }
#endif
}

#ifdef __HIP_PLATFORM_HCC__
template <>
void gemm<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  float falpha = alpha;
  float fbeta = beta;
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(at::BFloat16);
  TORCH_CUDABLAS_CHECK(rocblas_gemm_ex(
      handle,
      opa,
      opb,
      m,
      n,
      k,
      &falpha,
      a,
      rocblas_datatype_bf16_r,
      lda,
      b,
      rocblas_datatype_bf16_r,
      ldb,
      &fbeta,
      c,
      rocblas_datatype_bf16_r,
      ldc,
      c,
      rocblas_datatype_bf16_r,
      ldc,
      rocblas_datatype_f32_r,
      rocblas_gemm_algo_standard,
      0,
      0));
}
#endif


/* LEVEL 2 BLAS FUNCTIONS */

#define GEMV_CHECK_ARGVALUES(Dtype)           \
  do {                                        \
    CUDABLAS_NONNEGINT_CHECK(gemv<Dtype>, m); \
    CUDABLAS_NONNEGINT_CHECK(gemv<Dtype>, n); \
    CUDABLAS_POSINT_CHECK(gemv<Dtype>, lda);  \
    CUDABLAS_POSINT_CHECK(gemv<Dtype>, incx); \
    CUDABLAS_POSINT_CHECK(gemv<Dtype>, incy); \
  } while (0)

template <>
void gemv<double>(CUDABLAS_GEMV_ARGTYPES(double)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t op = _cublasOpFromChar(trans);
  _cublasAdjustLdLevel2(m, n, &lda);
  GEMV_CHECK_ARGVALUES(double);
  TORCH_CUDABLAS_CHECK(
      cublasDgemv(handle, op, m, n, &alpha, a, lda, x, incx, &beta, y, incy));
}

template <>
void gemv<float>(CUDABLAS_GEMV_ARGTYPES(float)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t op = _cublasOpFromChar(trans);
  _cublasAdjustLdLevel2(m, n, &lda);
  GEMV_CHECK_ARGVALUES(float);
  TORCH_CUDABLAS_CHECK(
      cublasSgemv(handle, op, m, n, &alpha, a, lda, x, incx, &beta, y, incy));
}

template <>
void gemv<at::Half>(CUDABLAS_GEMV_ARGTYPES(at::Half)) {
  TORCH_CHECK(
      incx == 1,
      "at::cuda::blas::gemv<Half>: support for incx != 1 not implemented");
  TORCH_CHECK(
      incy == 1,
      "at::cuda::blas::gemv<Half>: support for incy != 1 not implemented");
  gemm<at::Half>(
      trans, 'n', m, 1, n, alpha, a, n, x, n, beta, y, m);
}

#ifdef __HIP_PLATFORM_HCC__
template <>
void gemv<at::BFloat16>(CUDABLAS_GEMV_ARGTYPES(at::BFloat16)) {
  TORCH_CHECK(
      incx == 1,
      "at::cuda::blas::gemv<at::BFloat16>: support for incx != 1 not implemented");
  TORCH_CHECK(
      incy == 1,
      "at::cuda::blas::gemv<at::BFloat16>: support for incy != 1 not implemented");
  gemm<at::BFloat16>(
      trans, 'n', m, 1, n, alpha, a, n, x, n, beta, y, m);
}
#endif

} // namespace blas
} // namespace cuda
} // namespace at
