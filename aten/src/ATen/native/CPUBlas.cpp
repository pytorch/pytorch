#include <ATen/native/CPUBlas.h>
#include <ATen/Config.h>

#include <climits>

#if AT_BUILD_WITH_BLAS()
extern "C" void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, const double *a, int *lda, const double *b, int *ldb, double *beta, double *c, int *ldc);
extern "C" void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, const float *a, int *lda, const float *b, int *ldb, float *beta, float *c, int *ldc);
extern "C" void cgemm_(char *transa, char *transb, int *m, int *n, int *k, void *alpha, const void *a, int *lda, const void *b, int *ldb, void *beta, void *c, int *ldc);
extern "C" void zgemm_(char *transa, char *transb, int *m, int *n, int *k, void *alpha, const void *a, int *lda, const void *b, int *ldb, void *beta, void *c, int *ldc);
#endif  // AT_BUILD_WITH_BLAS()

#if AT_BUILD_WITH_BLAS()
extern "C" void cswap_(int *n, const void *x, int *incx, void *y, int *incy);
extern "C" void dcopy_(int *n, const double *x, int *incx, double *y, int *incy);
extern "C" void scopy_(int *n, const float *x, int *incx, float *y, int *incy);
extern "C" void zcopy_(int *n, const void *x, int *incx, void *y, int *incy);
extern "C" void ccopy_(int *n, const void *x, int *incx, void *y, int *incy);
extern "C" void daxpy_(int *n, double *a, const double *x, int *incx, double *y, int *incy);
extern "C" void saxpy_(int *n, float *a, const float *x, int *incx, float *y, int *incy);
extern "C" void caxpy_(int *n, void *a, const void *x, int *incx, void *y, int *incy);
extern "C" void zaxpy_(int *n, void *a, const void *x, int *incx, void *y, int *incy);
#endif  // AT_BUILD_WITH_BLAS()

#ifdef USE_FBGEMM
#include <fbgemm/FbgemmI64.h>
#endif  // USE_FBGEMM

namespace at {
namespace native {
namespace cpublas {
namespace internal {

void normalize_last_dims(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    int64_t *lda, int64_t *ldb, int64_t *ldc) {
  if (n == 1) {
    *ldc = m;
  }

  if(transa != NoTranspose) {
    if (m == 1) {
      *lda = k;
    }
  } else if(k == 1) {
    *lda = m;
  }

  if(transb != NoTranspose) {
    if (k == 1) {
      *ldb = n;
    }
  } else if (n == 1) {
    *ldb = k;
  }
}
}  // namespace internal

namespace {

bool use_blas_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    int64_t &lda, int64_t &ldb, int64_t &ldc) {
  const bool transa_ = transa != NoTranspose;
  const bool transb_ = transb != NoTranspose;
  return (
      (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) &&
      (lda <= INT_MAX) && (ldb <= INT_MAX) && (ldc <= INT_MAX) &&
      (lda >= std::max(int64_t{1}, (transa_ ? k : m))) &&
      (ldb >= std::max(int64_t{1}, (transb_ ? n : k))) &&
      (ldc >= std::max(int64_t{1}, m)));
}

#if AT_BUILD_WITH_BLAS()
char to_blas(TransposeType trans) {
  switch (trans) {
  case Transpose: return 't';
  case NoTranspose: return 'n';
  // case ConjTranspose: return 'c';
  }
  TORCH_INTERNAL_ASSERT(false, "Invalid transpose type");
}
#endif  // AT_BUILD_WITH_BLAS

#ifdef USE_FBGEMM
fbgemm::matrix_op_t to_fbgemm(TransposeType trans) {
  switch (trans) {
  case Transpose: return fbgemm::matrix_op_t::Transpose;
  case NoTranspose: return fbgemm::matrix_op_t::NoTranspose;
  // case ConjTranspose: return fbgemm::matrix_op_t::Transpose;
  }
  TORCH_INTERNAL_ASSERT(false, "Invalid transpose type");
}
#endif  // USE_FBGEMM

}  // namespace (anonymous)

DEFINE_DISPATCH(gemm_stub);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const double alpha,
    const double *a, int64_t lda,
    const double *b, int64_t ldb,
    const double beta,
    double *c, int64_t ldc) {
  internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
#if AT_BUILD_WITH_BLAS()
  if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
    int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
    char transa_ = to_blas(transa), transb_ = to_blas(transb);
    double alpha_ = alpha, beta_ = beta;
    dgemm_(
        &transa_, &transb_,
        &m_, &n_, &k_,
        &alpha_,
        a, &lda_,
        b, &ldb_,
        &beta_,
        c, &ldc_);
    return;
  }
#endif
  gemm_stub(
      at::kCPU, at::kDouble,
      transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const float alpha,
    const float *a, int64_t lda,
    const float *b, int64_t ldb,
    const float beta,
    float *c, int64_t ldc) {
  internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
#if AT_BUILD_WITH_BLAS()
  if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
    int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
    char transa_ = to_blas(transa), transb_ = to_blas(transb);
    float alpha_ = alpha, beta_ = beta;
    sgemm_(
        &transa_, &transb_,
        &m_, &n_, &k_,
        &alpha_,
        a, &lda_,
        b, &ldb_,
        &beta_,
        c, &ldc_);
    return;
  }
#endif
  gemm_stub(
      at::kCPU, at::kFloat,
      transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const c10::complex<double> alpha,
    const c10::complex<double> *a, int64_t lda,
    const c10::complex<double> *b, int64_t ldb,
    const c10::complex<double> beta,
    c10::complex<double> *c, int64_t ldc) {
  internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
#if AT_BUILD_WITH_BLAS()
  if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
    int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
    char transa_ = to_blas(transa), transb_ = to_blas(transb);
    c10::complex<double> alpha_ = alpha, beta_ = beta;
    zgemm_(
        &transa_, &transb_,
        &m_, &n_, &k_,
        &alpha_,
        a, &lda_,
        b, &ldb_,
        &beta_,
        c, &ldc_);
    return;
  }
#endif
  gemm_stub(
      at::kCPU, at::kComplexDouble,
      transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const c10::complex<float> alpha,
    const c10::complex<float> *a, int64_t lda,
    const c10::complex<float> *b, int64_t ldb,
    const c10::complex<float> beta,
    c10::complex<float> *c, int64_t ldc) {
  internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
#if AT_BUILD_WITH_BLAS()
  if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
    int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
    char transa_ = to_blas(transa), transb_ = to_blas(transb);
    c10::complex<float> alpha_ = alpha, beta_ = beta;
    cgemm_(
        &transa_, &transb_,
        &m_, &n_, &k_,
        &alpha_,
        a, &lda_,
        b, &ldb_,
        &beta_,
        c, &ldc_);
    return;
  }
#endif
  gemm_stub(
      at::kCPU, at::kComplexFloat,
      transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const int64_t alpha,
    const int64_t *a, int64_t lda,
    const int64_t *b, int64_t ldb,
    const int64_t beta,
    int64_t *c, int64_t ldc) {
  internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
#ifdef USE_FBGEMM
  if (alpha == 1 && (beta == 0 || beta == 1)) {
    // In FBGEMM, we assume row-major ordering; However, here we assume the
    // column-major ordering following the FORTRAN tradition in BLAS interface
    // in this function: we can configure the layout (row/column-major ordering)
    // of A and B by changing transa_ and transb_, but we cannot change the
    // layout of C with this FORTRAN-style BLAS interface.
    //
    // The workaround is that we compute
    // C^T (n x m) = B^T (n x k) * A^T (k x m) instead.
    //
    // In this way we view C^T as the row-major ordering when passing to FBGEMM.
    fbgemm::cblas_gemm_i64_i64acc(
        to_fbgemm(transb),
        to_fbgemm(transa),
        n,
        m,
        k,
        b,
        ldb,
        a,
        lda,
        beta == 1,
        c,
        ldc);
    return;
  }
#endif

  gemm_stub(
      kCPU, kLong,
      transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

DEFINE_DISPATCH(axpy_stub);

void axpy(int64_t n, double a, const double *x, int64_t incx, double *y, int64_t incy) {
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  #if AT_BUILD_WITH_BLAS()
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    daxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
    return;
  }
  #endif
  axpy_stub(
      kCPU, at::kDouble,
      n, a, x, incx, y, incy);
}

void axpy(int64_t n, float a, const float *x, int64_t incx, float *y, int64_t incy) {
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  #if AT_BUILD_WITH_BLAS()
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    saxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
    return;
  }
  #endif
  axpy_stub(
      kCPU, at::kFloat,
      n, a, x, incx, y, incy);
}

void axpy(int64_t n, c10::complex<double> a, const c10::complex<double> *x, int64_t incx, c10::complex<double> *y, int64_t incy) {
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  #if AT_BUILD_WITH_BLAS()
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    zaxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
    return;
  }
  #endif
  axpy_stub(
      kCPU, at::kComplexDouble,
      n, a, x, incx, y, incy);
}

void axpy(int64_t n, c10::complex<float> a, const c10::complex<float> *x, int64_t incx, c10::complex<float> *y, int64_t incy) {
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  #if AT_BUILD_WITH_BLAS()
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    caxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
    return;
  }
  #endif
  axpy_stub(
      kCPU, at::kComplexFloat,
      n, a, x, incx, y, incy);
}

DEFINE_DISPATCH(copy_stub);

void copy(int64_t n, const double *x, int64_t incx, double *y, int64_t incy) {
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  #if AT_BUILD_WITH_BLAS()
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) ) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    dcopy_(&i_n, x, &i_incx, y, &i_incy);
    return;
  }
  #endif
  copy_stub(
      kCPU, at::kDouble,
      n, x, incx, y, incy);
}

void copy(int64_t n, const float *x, int64_t incx, float *y, int64_t incy) {
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  #if AT_BUILD_WITH_BLAS()
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) ) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    scopy_(&i_n, x, &i_incx, y, &i_incy);
    return;
  }
  #endif
  copy_stub(
      kCPU, at::kFloat,
      n, x, incx, y, incy);
}

void copy(int64_t n, const c10::complex<double> *x, int64_t incx, c10::complex<double> *y, int64_t incy) {
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  #if AT_BUILD_WITH_BLAS()
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) ) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    zcopy_(&i_n, x, &i_incx, y, &i_incy);
    return;
  }
  #endif
  copy_stub(
      kCPU, at::kComplexDouble,
      n, x, incx, y, incy);
}

void copy(int64_t n, const c10::complex<float> *x, int64_t incx, c10::complex<float> *y, int64_t incy){
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  #if AT_BUILD_WITH_BLAS()
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) ) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    ccopy_(&i_n, x, &i_incx, y, &i_incy);
    return;
  }
  #endif
  copy_stub(
      kCPU, at::kComplexFloat,
      n, x, incx, y, incy);
}

}}}  // namespace at::native::cpublas
