#include <ATen/native/CPUBlas.h>
#include <ATen/Config.h>

#include <climits>

#if AT_MKL_ENABLED()
#  include "mkl_cblas.h"
#elif AT_BUILD_WITH_BLAS()
#  include "cblas.h"
#endif

#ifdef USE_FBGEMM
#include <fbgemm/FbgemmI64.h>
#endif  // USE_FBGEMM

namespace at {
namespace native {
namespace cpublas {
namespace {

void normalize_last_dims(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    int64_t &lda, int64_t &ldb, int64_t &ldc) {
  if (n == 1) {
    ldc = m;
  }

  if(transa != NoTranspose) {
    if(m == 1) {
      lda = k;
    }
  } else if(k == 1) {
    lda = m;
  }

  if(transb != NoTranspose) {
    if(k == 1) {
      ldb = n;
    }
  } else if (n == 1) {
    ldb = k;
  }
}

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
CBLAS_TRANSPOSE to_cblas(TransposeType trans) {
  switch (trans) {
  case Transpose: return CblasTrans;
  case NoTranspose: return CblasNoTrans;
  // case ConjTranspose: return CblasConjTrans;
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
#if AT_BUILD_WITH_BLAS()
  normalize_last_dims(transa, transb, m, n, k, lda, ldb, ldc);
  if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
    cblas_dgemm(
        CblasColMajor,
        to_cblas(transa), to_cblas(transb),
        m, n, k,
        alpha,
        a, lda,
        b, ldb,
        beta,
        c, ldc);
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
#if AT_BUILD_WITH_BLAS()
  normalize_last_dims(transa, transb, m, n, k, lda, ldb, ldc);
  if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
    cblas_sgemm(
        CblasColMajor,
        to_cblas(transa), to_cblas(transb),
        m, n, k,
        alpha,
        a, lda,
        b, ldb,
        beta,
        c, ldc);
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
#if AT_BUILD_WITH_BLAS()
  normalize_last_dims(transa, transb, m, n, k, lda, ldb, ldc);
  if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
    cblas_zgemm(
        CblasColMajor,
        to_cblas(transa), to_cblas(transb),
        m, n, k,
        &alpha,
        a, lda,
        b, ldb,
        &beta,
        c, ldc);
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
#if AT_BUILD_WITH_BLAS()
  normalize_last_dims(transa, transb, m, n, k, lda, ldb, ldc);
  if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
    cblas_cgemm(
        CblasColMajor,
        to_cblas(transa), to_cblas(transb),
        m, n, k,
        &alpha,
        a, lda,
        b, ldb,
        &beta,
        c, ldc);
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

}}}  // namespace at::native::cpublas
