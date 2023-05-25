#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/CPUBlas.h>
#include <ATen/native/mkl/LinearAlgebra.h>
#include <ATen/native/mkldnn/Matmul.h>
#include <ATen/Config.h>

#include <c10/util/SmallBuffer.h>
#include <c10/util/C++17.h>
#include <c10/util/irange.h>

#include <climits>

#if AT_BUILD_WITH_BLAS()
#if C10_IOS
#include <Accelerate/Accelerate.h>
#else
extern "C" void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, const double *a, int *lda, const double *b, int *ldb, double *beta, double *c, int *ldc);
extern "C" void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, const float *a, int *lda, const float *b, int *ldb, float *beta, float *c, int *ldc);
extern "C" void cgemm_(char *transa, char *transb, int *m, int *n, int *k, void *alpha, const void *a, int *lda, const void *b, int *ldb, void *beta, void *c, int *ldc);
extern "C" void zgemm_(char *transa, char *transb, int *m, int *n, int *k, void *alpha, const void *a, int *lda, const void *b, int *ldb, void *beta, void *c, int *ldc);
#ifdef BLAS_HAS_SBGEMM
extern "C" void sbgemm_(char *transa, char *transb, int *m, int *n, int *k,
                float *alpha,
                const at::BFloat16 *a, int *lda,
                const at::BFloat16 *b, int *ldb,
                float *beta,
                float *c, int *ldc);
#endif  // BLAS_HAS_SBGEMM
extern "C" void cswap_(int *n, const void *x, int *incx, void *y, int *incy);
extern "C" void dcopy_(int *n, const double *x, int *incx, double *y, int *incy);
extern "C" void scopy_(int *n, const float *x, int *incx, float *y, int *incy);
extern "C" void zcopy_(int *n, const void *x, int *incx, void *y, int *incy);
extern "C" void ccopy_(int *n, const void *x, int *incx, void *y, int *incy);
extern "C" void daxpy_(int *n, double *a, const double *x, int *incx, double *y, int *incy);
extern "C" void saxpy_(int *n, float *a, const float *x, int *incx, float *y, int *incy);
extern "C" void caxpy_(int *n, void *a, const void *x, int *incx, void *y, int *incy);
extern "C" void zaxpy_(int *n, void *a, const void *x, int *incx, void *y, int *incy);
#endif  // C10_IOS
#endif  // AT_BUILD_WITH_BLAS

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

  if(transa != TransposeType::NoTranspose) {
    if (m == 1) {
      *lda = k;
    }
  } else if(k == 1) {
    *lda = m;
  }

  if(transb != TransposeType::NoTranspose) {
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
    int64_t lda, int64_t ldb, int64_t ldc) {
  const bool transa_ = transa != TransposeType::NoTranspose;
  const bool transb_ = transb != TransposeType::NoTranspose;
  return (
      (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) &&
      (lda <= INT_MAX) && (ldb <= INT_MAX) && (ldc <= INT_MAX) &&
      (lda >= std::max(int64_t{1}, (transa_ ? k : m))) &&
      (ldb >= std::max(int64_t{1}, (transb_ ? n : k))) &&
      (ldc >= std::max(int64_t{1}, m)));
}

#ifdef USE_FBGEMM
fbgemm::matrix_op_t to_fbgemm(TransposeType trans) {
  switch (trans) {
    case TransposeType::Transpose: return fbgemm::matrix_op_t::Transpose;
    case TransposeType::NoTranspose: return fbgemm::matrix_op_t::NoTranspose;
    case TransposeType::ConjTranspose: TORCH_INTERNAL_ASSERT(false, "ConjTranspose type is not supported in fbgemm");
  }
  TORCH_INTERNAL_ASSERT(false, "Invalid transpose type");
}
#endif  // USE_FBGEMM

#if (AT_BUILD_WITH_BLAS() && C10_IOS)
CBLAS_TRANSPOSE to_apple_accelerate_transpose(TransposeType trans) {
  switch (trans) {
    case TransposeType::Transpose: return CblasTrans;
    case TransposeType::NoTranspose: return CblasNoTrans;
    case TransposeType::ConjTranspose: return CblasConjTrans;
  }
  TORCH_INTERNAL_ASSERT(false, "Invalid transpose type");
}
#endif

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
    double alpha_ = alpha, beta_ = beta;
    #if C10_IOS
    CBLAS_TRANSPOSE transa_ = to_apple_accelerate_transpose(transa);
    CBLAS_TRANSPOSE transb_ = to_apple_accelerate_transpose(transb);
    cblas_dgemm(CblasColMajor,
      transa_, transb_,
      m_, n_, k_,
      alpha_,
      a, lda_,
      b, ldb_,
      beta_,
      c, ldc_);
    #else
    char transa_ = to_blas(transa), transb_ = to_blas(transb);
    dgemm_(
        &transa_, &transb_,
        &m_, &n_, &k_,
        &alpha_,
        a, &lda_,
        b, &ldb_,
        &beta_,
        c, &ldc_);
    #endif
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
    float alpha_ = alpha, beta_ = beta;
    #if C10_IOS
    CBLAS_TRANSPOSE transa_ = to_apple_accelerate_transpose(transa);
    CBLAS_TRANSPOSE transb_ = to_apple_accelerate_transpose(transb);
    cblas_sgemm(CblasColMajor,
      transa_, transb_,
      m_, n_, k_,
      alpha_,
      a, lda_,
      b, ldb_,
      beta_,
      c, ldc_);
    #else
    char transa_ = to_blas(transa), transb_ = to_blas(transb);
    sgemm_(
        &transa_, &transb_,
        &m_, &n_, &k_,
        &alpha_,
        a, &lda_,
        b, &ldb_,
        &beta_,
        c, &ldc_);
    #endif
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
    c10::complex<double> alpha_ = alpha, beta_ = beta;
    #if C10_IOS
    CBLAS_TRANSPOSE transa_ = to_apple_accelerate_transpose(transa);
    CBLAS_TRANSPOSE transb_ = to_apple_accelerate_transpose(transb);
    cblas_zgemm(CblasColMajor,
      transa_, transb_,
      m_, n_, k_,
      &alpha_,
      a, lda_,
      b, ldb_,
      &beta_,
      c, ldc_);
    #else
    char transa_ = to_blas(transa), transb_ = to_blas(transb);
    zgemm_(
        &transa_, &transb_,
        &m_, &n_, &k_,
        &alpha_,
        a, &lda_,
        b, &ldb_,
        &beta_,
        c, &ldc_);
    #endif
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
    c10::complex<float> alpha_ = alpha, beta_ = beta;
    #if C10_IOS
    CBLAS_TRANSPOSE transa_ = to_apple_accelerate_transpose(transa);
    CBLAS_TRANSPOSE transb_ = to_apple_accelerate_transpose(transb);
    cblas_cgemm(CblasColMajor,
      transa_, transb_,
      m_, n_, k_,
      &alpha_,
      a, lda_,
      b, ldb_,
      &beta_,
      c, ldc_);
    #else
    char transa_ = to_blas(transa), transb_ = to_blas(transb);
    cgemm_(
        &transa_, &transb_,
        &m_, &n_, &k_,
        &alpha_,
        a, &lda_,
        b, &ldb_,
        &beta_,
        c, &ldc_);
    #endif
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
   const float alpha,
   const at::BFloat16 *a, int64_t lda,
   const at::BFloat16 *b, int64_t ldb,
   const float beta,
   at::BFloat16 *c, int64_t ldc) {
   internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
#if AT_BUILD_WITH_BLAS() && defined(BLAS_HAS_SBGEMM)
   if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
      int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
      char transa_ = to_blas(transa), transb_ = to_blas(transb);
      float alpha_ = alpha, beta_ = beta;
      int c_size = n_ * ldc_;
      // C matrix in OpenBLAS sbgemm are of type "float" so we have to convert, copy and copy back.
      std::vector<float> float_v(c, c + c_size);
      sbgemm_(&transa_, &transb_,
              &m_, &n_, &k_,
              &alpha_,
              a, &lda_,
              b, &ldb_,
              &beta_,
              float_v.data(), &ldc_);
      for (auto cv: float_v) {
        *(c++) = c10::convert<at::BFloat16>(cv);
      }
      return;
   }
#endif
#if AT_MKLDNN_ENABLED()
   if (mkldnn_bf16_gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) {
     return;
   }
#endif
   gemm_stub(
      at::kCPU, at::kBFloat16,
      transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(
   TransposeType transa, TransposeType transb,
   int64_t m, int64_t n, int64_t k,
   const float alpha,
   const at::Half *a, int64_t lda,
   const at::Half *b, int64_t ldb,
   const float beta,
   at::Half *c, int64_t ldc) {
   internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
#if AT_MKLDNN_ENABLED()
   if (mkldnn_fp16_gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) {
     return;
   }
#endif
   gemm_stub(
      at::kCPU, at::kHalf,
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

template <typename scalar_t>
static void gemm_batched_mkl_impl(
      TransposeType transa, TransposeType transb,
      int64_t batch_size, int64_t m, int64_t n, int64_t k,
      scalar_t alpha,
      const scalar_t **a, int64_t lda,
      const scalar_t **b, int64_t ldb,
      scalar_t beta,
      scalar_t **c, int64_t ldc) {
  for (int64_t i = 0; i < batch_size;) {
    int sub_batch = std::min(batch_size - i, int64_t{INT_MAX});
    mkl_gemm_batched(transa, transb, sub_batch, m, n, k, alpha,
                     &a[i], lda, &b[i], ldb, beta, &c[i], ldc);
    i += sub_batch;
  }
}

template <typename scalar_t>
using is_blas_library_type = std::integral_constant<bool,
    std::is_same<scalar_t, double>::value ||
    std::is_same<scalar_t, float>::value ||
    std::is_same<scalar_t, c10::complex<double>>::value ||
    std::is_same<scalar_t, c10::complex<float>>::value>;

template <typename scalar_t>
void gemm_batched_generic(
    TransposeType transa, TransposeType transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t **a, int64_t lda,
    const scalar_t **b, int64_t ldb,
    scalar_t beta,
    scalar_t **c, int64_t ldc) {
  for (const auto batch : c10::irange(batch_size)) {
    gemm(transa, transb, m, n, k, alpha, a[batch], lda, b[batch], ldb, beta, c[batch], ldc);
  }
}

template <typename scalar_t>
void gemm_batched(
    TransposeType transa, TransposeType transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t **a, int64_t lda,
    const scalar_t **b, int64_t ldb,
    scalar_t beta,
    scalar_t **c, int64_t ldc) {
  if (batch_size == 1) {
    return gemm(transa, transb, m, n, k, alpha, a[0], lda, b[0], ldb, beta, c[0], ldc);
  }

  if constexpr (AT_MKL_ENABLED() && is_blas_library_type<scalar_t>::value) {
    internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
    if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
      gemm_batched_mkl_impl(
          transa, transb, batch_size, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else {
      gemm_batched_generic(
          transa, transb, batch_size, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
  } else {
    gemm_batched_generic(
        transa, transb, batch_size, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}

template <typename scalar_t>
void gemm_batched_with_stride_generic(
    TransposeType transa, TransposeType transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t *a, int64_t lda, int64_t batch_stride_a,
    const scalar_t *b, int64_t ldb, int64_t batch_stride_b,
    scalar_t beta,
    scalar_t *c, int64_t ldc, int64_t batch_stride_c) {
  for (const auto batch : c10::irange(batch_size)) {
    const auto a_batch = a + batch_stride_a * batch;
    const auto b_batch = b + batch_stride_b * batch;
    const auto c_batch = c + batch_stride_c * batch;
    gemm(transa, transb, m, n, k, alpha, a_batch, lda, b_batch, ldb, beta, c_batch, ldc);
  }
}

template <typename scalar_t>
void gemm_batched_with_stride(
    TransposeType transa, TransposeType transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t *a, int64_t lda, int64_t batch_stride_a,
    const scalar_t *b, int64_t ldb, int64_t batch_stride_b,
    scalar_t beta,
    scalar_t *c, int64_t ldc, int64_t batch_stride_c) {
  if (batch_size == 1) {
    return gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }

  if constexpr (AT_MKL_ENABLED() && is_blas_library_type<scalar_t>::value) {
    internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
    if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
      c10::SmallBuffer<const scalar_t*, 16> a_ptrs(batch_size);
      c10::SmallBuffer<const scalar_t*, 16> b_ptrs(batch_size);
      c10::SmallBuffer<scalar_t*, 16> c_ptrs(batch_size);

      for (const auto batch : c10::irange(batch_size)) {
        a_ptrs[batch] = a + batch_stride_a * batch;
        b_ptrs[batch] = b + batch_stride_b * batch;
        c_ptrs[batch] = c + batch_stride_c * batch;
      }
      gemm_batched_mkl_impl(
          transa, transb, batch_size, m, n, k, alpha, a_ptrs.data(), lda,
          b_ptrs.data(), ldb, beta, c_ptrs.data(), ldc);
    } else {
      gemm_batched_with_stride_generic(
          transa, transb, batch_size, m, n, k, alpha, a, lda, batch_stride_a,
          b, ldb, batch_stride_b, beta, c, ldc, batch_stride_c);
    }
  } else {
    gemm_batched_with_stride_generic(transa, transb, batch_size, m, n, k, alpha,
                                     a, lda, batch_stride_a, b, ldb, batch_stride_b,
                                     beta, c, ldc, batch_stride_c);
  }
}

#define INSTANTIATE_BATCHED_GEMM(scalar_t, DType)               \
  template void gemm_batched(                                   \
      TransposeType transa, TransposeType transb,               \
      int64_t batch_size, int64_t m, int64_t n, int64_t k,      \
      scalar_t alpha,                                           \
      const scalar_t **a, int64_t lda,                          \
      const scalar_t **b, int64_t ldb,                          \
      scalar_t beta,                                            \
      scalar_t **c, int64_t ldc);                               \
  template void gemm_batched_with_stride(                       \
      TransposeType transa, TransposeType transb,               \
      int64_t batch_size, int64_t m, int64_t n, int64_t k,      \
      scalar_t alpha,                                           \
      const scalar_t *a, int64_t lda, int64_t batch_stride_a,   \
      const scalar_t *b, int64_t ldb, int64_t batch_stride_b,   \
      scalar_t beta,                                            \
      scalar_t *c, int64_t ldc, int64_t batch_stride_c);

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(INSTANTIATE_BATCHED_GEMM)

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
    #if C10_IOS
    cblas_daxpy(i_n, a, x, i_incx, y, i_incy);
    #else
    daxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
    #endif
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
    #if C10_IOS
    cblas_saxpy(i_n, a, x, i_incx, y, i_incy);
    #else
    saxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
    #endif
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
    #if C10_IOS
    cblas_zaxpy(i_n, &a, x, i_incx, y, i_incy);
    #else
    zaxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
    #endif
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
    #if C10_IOS
    cblas_caxpy(i_n, &a, x, i_incx, y, i_incy);
    #else
    caxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
    #endif
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
    #if C10_IOS
    cblas_dcopy(i_n, x, i_incx, y, i_incy);
    #else
    dcopy_(&i_n, x, &i_incx, y, &i_incy);
    #endif
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
    #if C10_IOS
    cblas_scopy(i_n, x, i_incx, y, i_incy);
    #else
    scopy_(&i_n, x, &i_incx, y, &i_incy);
    #endif
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
    #if C10_IOS
    cblas_zcopy(i_n, x, i_incx, y, i_incy);
    #else
    zcopy_(&i_n, x, &i_incx, y, &i_incy);
    #endif
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
    #if C10_IOS
    cblas_ccopy(i_n, &x, i_incx, y, i_incy);
    #else
    ccopy_(&i_n, x, &i_incx, y, &i_incy);
    #endif
    return;
  }
  #endif
  copy_stub(
      kCPU, at::kComplexFloat,
      n, x, incx, y, incy);
}

}}}  // namespace at::native::cpublas
