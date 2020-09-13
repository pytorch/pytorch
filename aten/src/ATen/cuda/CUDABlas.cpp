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
  globalContext().alertCuBLASConfigNotDeterministic();
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
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(float);
  TORCH_CUDABLAS_CHECK(cublasSgemm(
      handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}

#if !defined(__HIP_PLATFORM_HCC__) || (defined(__HIP_PLATFORM_HCC__) && HIP_VERSION >= 210)
  template <>
  void gemm<c10::complex<double>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<double>)) {
    globalContext().alertCuBLASConfigNotDeterministic();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasOperation_t opa = _cublasOpFromChar(transa);
    cublasOperation_t opb = _cublasOpFromChar(transb);
    _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
    GEMM_CHECK_ARGVALUES(c10::complex<double>);
    TORCH_CUDABLAS_CHECK(cublasZgemm(
        handle, opa, opb, m, n, k, reinterpret_cast<const cuDoubleComplex*>(&alpha), reinterpret_cast<const cuDoubleComplex*>(a),
        lda, reinterpret_cast<const cuDoubleComplex*>(b), ldb, reinterpret_cast<const cuDoubleComplex*>(&beta),
        reinterpret_cast<cuDoubleComplex*>(c), ldc));
  }
#endif

#if !defined(__HIP_PLATFORM_HCC__) || (defined(__HIP_PLATFORM_HCC__) && HIP_VERSION >= 210)
  template <>
  void gemm<c10::complex<float>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<float>)) {
    globalContext().alertCuBLASConfigNotDeterministic();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasOperation_t opa = _cublasOpFromChar(transa);
    cublasOperation_t opb = _cublasOpFromChar(transb);
    _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
    GEMM_CHECK_ARGVALUES(c10::complex<float>);
    TORCH_CUDABLAS_CHECK(cublasCgemm(
        handle, opa, opb, m, n, k, reinterpret_cast<const cuComplex*>(&alpha), reinterpret_cast<const cuComplex*>(a),
        lda, reinterpret_cast<const cuComplex*>(b), ldb, reinterpret_cast<const cuComplex*>(&beta),
        reinterpret_cast<cuComplex*>(c), ldc));
  }
#endif

template <>
void gemm<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half)) {
  globalContext().alertCuBLASConfigNotDeterministic();
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
#if defined(CUDA_VERSION) && CUDA_VERSION < 11000
    // On CUDA versions prior to 11, users are required to set the math mode to CUBLAS_TENSOR_OP_MATH
    // manually to be able to use tensor cores for FP16. On CUDA 11, this is no longer required.
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
#endif  // CUDA_VERSION < 11000
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
#if defined(CUDA_VERSION) && CUDA_VERSION < 11000
    // On CUDA versions prior to 11, users are required to set the math mode to CUBLAS_TENSOR_OP_MATH
    // manually to be able to use tensor cores for FP16. On CUDA 11, this is no longer required.
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
#endif  // CUDA_VERSION < 11000
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

#if !defined(__HIP_PLATFORM_HCC__) || (defined(__HIP_PLATFORM_HCC__) && HIP_VERSION >= 210)
  template <>
  void gemv<c10::complex<double>>(CUDABLAS_GEMV_ARGTYPES(c10::complex<double>)) {
    globalContext().alertCuBLASConfigNotDeterministic();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasOperation_t op = _cublasOpFromChar(trans);
    _cublasAdjustLdLevel2(m, n, &lda);
    GEMV_CHECK_ARGVALUES(c10::complex<double>);
    TORCH_CUDABLAS_CHECK(
        cublasZgemv(handle, op, m, n, reinterpret_cast<const cuDoubleComplex*>(&alpha), reinterpret_cast<const cuDoubleComplex*>(a),
        lda, reinterpret_cast<const cuDoubleComplex*>(x), incx, reinterpret_cast<const cuDoubleComplex*>(&beta),
        reinterpret_cast<cuDoubleComplex*>(y), incy));
  }
#endif

#if !defined(__HIP_PLATFORM_HCC__) || (defined(__HIP_PLATFORM_HCC__) && HIP_VERSION >= 210)
  template <>
  void gemv<c10::complex<float>>(CUDABLAS_GEMV_ARGTYPES(c10::complex<float>)) {
    globalContext().alertCuBLASConfigNotDeterministic();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasOperation_t op = _cublasOpFromChar(trans);
    _cublasAdjustLdLevel2(m, n, &lda);
    GEMV_CHECK_ARGVALUES(c10::complex<float>);
    TORCH_CUDABLAS_CHECK(
        cublasCgemv(handle, op, m, n, reinterpret_cast<const cuComplex*>(&alpha), reinterpret_cast<const cuComplex*>(a),
        lda, reinterpret_cast<const cuComplex*>(x), incx, reinterpret_cast<const cuComplex*>(&beta),
        reinterpret_cast<cuComplex*>(y), incy));
  }
#endif

template <>
void gemv<double>(CUDABLAS_GEMV_ARGTYPES(double)) {
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t op = _cublasOpFromChar(trans);
  _cublasAdjustLdLevel2(m, n, &lda);
  GEMV_CHECK_ARGVALUES(double);
  TORCH_CUDABLAS_CHECK(
      cublasDgemv(handle, op, m, n, &alpha, a, lda, x, incx, &beta, y, incy));
}

template <>
void gemv<float>(CUDABLAS_GEMV_ARGTYPES(float)) {
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t op = _cublasOpFromChar(trans);
  _cublasAdjustLdLevel2(m, n, &lda);
  GEMV_CHECK_ARGVALUES(float);
  TORCH_CUDABLAS_CHECK(
      cublasSgemv(handle, op, m, n, &alpha, a, lda, x, incx, &beta, y, incy));
}

template <>
void gemv<at::Half>(CUDABLAS_GEMV_ARGTYPES(at::Half)) {
  // In general, cublas regards matrices as column-major.
  // The cublasS/Dgemv usages in cuda::blas::gemv<float>/<double> above
  // require that external blas::gemv callers obey the following convention:
  //
  // If "a" is row-major with shape (output, summed) in blas::gemv's caller,
  // caller interprets it as column-major with shape (summed, output), passes
  // summed and output respectively to our local vars m, n, and requests that cublas
  // internally transpose ("trans") the column-major interpretation of a.
  //
  // There's no such thing as "cublasHalfgemv", so here we hack gemv with a gemm.
  // However, we must allow the same calling convention, because the caller shouldn't
  // have to swap args based on whether it's calling blas::gemv<at::Half> or <float>.

  bool trans_bool = (_cublasOpFromChar(trans) != CUBLAS_OP_N);
  if (trans_bool) {
    std::swap(m, n);
  }
  // After swap, local vars m, n contain the output and summed sizes respectively,
  // regardless of whether "a" was row-major or column-major in gemv<>'s caller.

  // To handle the possibility incy > 1, interprets vector y as column-major matrix with one row
  // (shape (1, output)) and leading dim incy.
  // trans(a)*x would compute a matrix with one column (shape (output, 1)) which wouldn't match y.
  // So instead, we interpret x similarly to y, as a column-major matrix with one row
  // (shape (1, summed)) and leading dim incx.  The gemm then carries out x*transpose(trans(a)) to
  // produce a matrix with one row (shape (1, output)), matching y.
  char trans_flipped = (trans_bool ? 'n' : 't');
  gemm<at::Half>(
      'n', trans_flipped, 1, m, n, alpha, x, incx, a, lda, beta, y, incy);
}

#ifdef __HIP_PLATFORM_HCC__
template <>
void gemv<at::BFloat16>(CUDABLAS_GEMV_ARGTYPES(at::BFloat16)) {
  bool trans_bool = (_cublasOpFromChar(trans) != CUBLAS_OP_N);
  if (trans_bool) {
    std::swap(m, n);
  }
  char trans_flipped = (trans_bool ? 'n' : 't');
  gemm<at::BFloat16>(
      'n', trans_flipped, 1, m, n, alpha, x, incx, a, lda, beta, y, incy);
}
#endif

namespace {
template<typename scalar_t>
cublasStatus_t cublasGer(const cublasHandle_t &handle, int64_t m, int64_t n, scalar_t *alpha, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy, scalar_t *a, int64_t lda) {
  TORCH_CHECK(false, "cublas ger is defined only for float and double");
  return {};
}
template<>
cublasStatus_t cublasGer<float>(const cublasHandle_t &handle, int64_t m, int64_t n, float *alpha, float *x, int64_t incx, float *y, int64_t incy, float *a, int64_t lda) {
  return cublasSger(handle, m, n, alpha, x, incx, y, incy, a, lda);
}
template<>
cublasStatus_t cublasGer<double>(const cublasHandle_t &handle, int64_t m, int64_t n, double *alpha, double *x, int64_t incx, double *y, int64_t incy, double *a, int64_t lda) {
  return cublasDger(handle, m, n, alpha, x, incx, y, incy, a, lda);
}
} // anonymous namespace

template<typename scalar_t>
void ger(int64_t m, int64_t n, scalar_t alpha, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy, scalar_t *a, int64_t lda)
{
  _cublasAdjustLdLevel2(m, n, &lda);
  TORCH_CHECK((m <= INT_MAX) &&
              (n <= INT_MAX) &&
              (lda <= INT_MAX) &&
              (incx <= INT_MAX) &&
              (incy <= INT_MAX),
              "cublasSger/cublasDger only supports m, n, lda, incx, incy with "
              "the bound [val] <= %d", INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_lda = (int)lda;
  int i_incx = (int)incx;
  int i_incy = (int)incy;

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(cublasGer<scalar_t>(
    handle, i_m, i_n, &alpha, x, i_incx, y, i_incy, a, i_lda));
}
template void ger<float>(int64_t m, int64_t n, float alpha, float *x, int64_t incx, float *y, int64_t incy, float *a, int64_t lda);
template void ger<double>(int64_t m, int64_t n, double alpha, double *x, int64_t incx, double *y, int64_t incy, double *a, int64_t lda);

/* LEVEL 1 BLAS FUNCTIONS */

template <>
void dot<double>(CUDABLAS_DOT_ARGTYPES(double)) {
  TORCH_CUDABLAS_CHECK(cublasDdot(handle, n, x, incx, y, incy, result));
}

template <>
void dot<float>(CUDABLAS_DOT_ARGTYPES(float)) {
  TORCH_CUDABLAS_CHECK(cublasSdot(handle, n, x, incx, y, incy, result));
}

template <>
void dot<c10::complex<double>>(CUDABLAS_DOT_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDABLAS_CHECK(cublasZdotu(handle, n, reinterpret_cast<const cuDoubleComplex*>(x),
                                   incx, reinterpret_cast<const cuDoubleComplex*>(y), incy,
                                   reinterpret_cast<cuDoubleComplex*>(result)));
}

template <>
void dot<c10::complex<float>>(CUDABLAS_DOT_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDABLAS_CHECK(cublasCdotu(handle, n, reinterpret_cast<const cuComplex*>(x),
                                   incx, reinterpret_cast<const cuComplex*>(y), incy,
                                   reinterpret_cast<cuComplex*>(result)));
}

template <>
void dot<at::Half>(CUDABLAS_DOT_ARGTYPES(at::Half)) {
#if CUDA_VERSION >= 8000
  TORCH_CUDABLAS_CHECK(cublasDotEx(
      handle,
      n,
      x,
      CUDA_R_16F,
      incx,
      y,
      CUDA_R_16F,
      incy,
      result,
      CUDA_R_16F,
      CUDA_R_32F));
#elif HIP_VERSION >= 210
  TORCH_CUDABLAS_CHECK(rocblas_hdot(
      handle,
      n,
      reinterpret_cast<const rocblas_half*>(x),
      incx,
      reinterpret_cast<const rocblas_half*>(y),
      incy,
      reinterpret_cast<rocblas_half*>(result)));
#else
  AT_ERROR("Cublas_Hdot requires CUDA 8.0+");
#endif
}

template <>
void vdot<c10::complex<float>>(CUDABLAS_DOT_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDABLAS_CHECK(cublasCdotc(handle, n, reinterpret_cast<const cuComplex*>(x),
                                   incx, reinterpret_cast<const cuComplex*>(y), incy,
                                   reinterpret_cast<cuComplex*>(result)));
}

template <>
void vdot<c10::complex<double>>(CUDABLAS_DOT_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDABLAS_CHECK(cublasZdotc(handle, n, reinterpret_cast<const cuDoubleComplex*>(x),
                                   incx, reinterpret_cast<const cuDoubleComplex*>(y), incy,
                                   reinterpret_cast<cuDoubleComplex*>(result)));
}

} // namespace blas
} // namespace cuda
} // namespace at
