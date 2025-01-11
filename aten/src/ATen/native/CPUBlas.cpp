#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/CPUBlas.h>
#include <ATen/native/mkl/LinearAlgebra.h>
#include <ATen/native/onednn/Matmul.h>
#include <ATen/Config.h>

#include <c10/util/SmallBuffer.h>
#include <c10/util/irange.h>

#include <climits>
#if !defined(__s390x__ ) && !defined(__powerpc__)
#include <cpuinfo.h>
#endif

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

#if AT_MKLDNN_ENABLED()
#include <ideep.hpp>
// Add uKernel API versioning to be compatible with different oneDNN versions
// oneDNN 3.6.x updates the ukernel APIs of brgemm and brgemm_pack_B
// brgemm_pack_B is changed to transform and the setting of brgemm beta is changed to set_add_C
#if (IDEEP_VERSION_MAJOR == 3 && IDEEP_VERSION_MINOR == 5)
#define ONEDNN_UKERNEL_1
#elif (IDEEP_VERSION_MAJOR >= 3 && IDEEP_VERSION_MINOR >= 6)
#define ONEDNN_UKERNEL_2
#endif
#if ((defined(ONEDNN_UKERNEL_1) || defined(ONEDNN_UKERNEL_2)) && (defined(__x86_64__) || (defined(_M_X64) && !defined(_M_ARM64EC))))
#define ONEDNN_UKERNEL_ENABLED
#endif
#endif  // AT_MKLDNN_ENABLED()

#if defined(ONEDNN_UKERNEL_ENABLED)
#include <oneapi/dnnl/dnnl_ukernel.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#endif // oneDNN BRGEMM

namespace at::native::cpublas {
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
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunneeded-internal-declaration")
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
C10_DIAGNOSTIC_POP()

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
#if AT_MKLDNN_ENABLED()
   if (mkldnn_bf32_gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) {
     return;
   }
#endif
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
#ifdef __aarch64__
   // MKLDNN also supports ARM for bf16, and the bypass is only
   // currently intended for x86/x86_64.
   const bool use_bf16_gemv_trans = false;
#else
   const bool bf16_gemv_trans_would_be_faster = cpuinfo_initialize() &&
     !cpuinfo_has_x86_avx512bf16();
   const bool use_bf16_gemv_trans = bf16_gemv_trans_would_be_faster &&
     transa == TransposeType::Transpose &&
     transb == TransposeType::NoTranspose && n == 1 && alpha == 1.0;
#endif
   if (!use_bf16_gemv_trans && mkldnn_bf16_gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) {
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
   // Per https://github.com/pytorch/pytorch/pull/137918#discussion_r1825460179 ,
   // we should not bother checking for !cpuinfo_has_x86_avx512fp16() here,
   // because "onednn (mkldnn) won't use avx512fp16 to compute gemms by default
   // because the avx512fp16 fma would incur accuracy loss".
   const bool fp16_gemv_trans_would_be_faster = cpuinfo_initialize() &&
     cpuinfo_has_x86_f16c();
   const bool use_fp16_gemv_trans = fp16_gemv_trans_would_be_faster &&
     transa == TransposeType::Transpose &&
     transb == TransposeType::NoTranspose && n == 1 && alpha == 1.0;
   if (!use_fp16_gemv_trans &&
       mkldnn_fp16_gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) {
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
    const float alpha,
    const at::BFloat16 *a, int64_t lda,
    const at::BFloat16 *b, int64_t ldb,
    const float beta,
    float *c, int64_t ldc) {
  internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
#if AT_BUILD_WITH_BLAS() && defined(BLAS_HAS_SBGEMM)
   if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
      int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
      char transa_ = to_blas(transa), transb_ = to_blas(transb);
      float alpha_ = alpha, beta_ = beta;
      sbgemm_(&transa_, &transb_,
              &m_, &n_, &k_,
              &alpha_,
              a, &lda_,
              b, &ldb_,
              &beta_,
              c, &ldc_);
      return;
   }
#endif
#ifdef MKL_HAS_SBGEMM
  if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
    int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
    mkl_gemm_bf16bf16f32(transa, transb, m_, n_, k_, alpha, a, lda_, b, ldb_, beta, c, ldc_);
    return;
  }
#endif
  // for the fallback path, first compute gemm with beta = 0,
  // and then add c in full precision.
  int64_t c_size = n * m;
  std::vector<at::BFloat16> bfloat_c(c_size, 0.f);
  gemm_stub(
      at::kCPU, at::kBFloat16,
      transa, transb, m, n, k, alpha, a, lda, b, ldb, 0.f, bfloat_c.data(), m);
  for (const auto j : c10::irange(n)) {
    for (const auto i : c10::irange(m)) {
      auto offset = j * ldc + i;
      // beta == 0 won't propagate NaN from C
      if (beta == 0.f) {
        c[offset] = c10::convert<float>(bfloat_c[j * m + i]);
      } else {
        c[offset] = beta * c[offset] + c10::convert<float>(bfloat_c[j * m + i]);
      }
    }
  }
}

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const float alpha,
    const at::Half *a, int64_t lda,
    const at::Half *b, int64_t ldb,
    const float beta,
    float *c, int64_t ldc) {
  internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
#ifdef MKL_HAS_SHGEMM
  if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
    int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
    mkl_gemm_f16f16f32(transa, transb, m_, n_, k_, alpha, a, lda_, b, ldb_, beta, c, ldc_);
    return;
  }
#endif
  // for the fallback path, first compute gemm with beta = 0,
  // and then add c in full precision.
  int64_t c_size = n * m;
  std::vector<at::Half> float16_c(c_size, 0.f);
  gemm_stub(
      at::kCPU, at::kHalf,
      transa, transb, m, n, k, alpha, a, lda, b, ldb, 0.f, float16_c.data(), m);
  for (const auto j : c10::irange(n)) {
    for (const auto i : c10::irange(m)) {
      auto offset = j * ldc + i;
      // beta == 0 won't propagate NaN from C
      if (beta == 0.f) {
        c[offset] = c10::convert<float>(float16_c[j * m + i]);
      } else {
        c[offset] = beta * c[offset] + c10::convert<float>(float16_c[j * m + i]);
      }
    }
  }
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
    std::is_same_v<scalar_t, double> ||
    std::is_same_v<scalar_t, float> ||
    std::is_same_v<scalar_t, c10::complex<double>> ||
    std::is_same_v<scalar_t, c10::complex<float>>>;

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

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF_F8NZ(INSTANTIATE_BATCHED_GEMM)

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

// oneDNN BRGEMM
#if defined(ONEDNN_UKERNEL_ENABLED)
struct BrgemmKey {
  int64_t M;
  int64_t N;
  int64_t K;
  int64_t batch_size;
  int64_t lda;
  int64_t ldb;
  int64_t ldc;
  ScalarType dt_a;
  ScalarType dt_b;
  ScalarType dt_c;
  bool add_C;

  BrgemmKey(
      int64_t M,
      int64_t N,
      int64_t K,
      int64_t batch_size,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      ScalarType dt_a,
      ScalarType dt_b,
      ScalarType dt_c,
      bool add_C)
      : M(M),
        N(N),
        K(K),
        batch_size(batch_size),
        lda(lda),
        ldb(ldb),
        ldc(ldc),
        dt_a(dt_a),
        dt_b(dt_b),
        dt_c(dt_c),
        add_C(add_C) {}
  bool operator==(const BrgemmKey& other) const {
    return M == other.M && N == other.N && K == other.K &&
        batch_size == other.batch_size && lda == other.lda &&
        ldb == other.ldb && ldc == other.ldc && dt_a == other.dt_a &&
        dt_b == other.dt_b && dt_c == other.dt_c && add_C == other.add_C;
  }
};

struct PackKey {
  int64_t K;
  int64_t N;
  int64_t ld_in;
  int64_t ld_out;
  ScalarType dt_in;
  ScalarType dt_out;
  PackKey(
      int64_t K,
      int64_t N,
      int64_t ld_in,
      int64_t ld_out,
      ScalarType dt_in,
      ScalarType dt_out)
      : K(K),
        N(N),
        ld_in(ld_in),
        ld_out(ld_out),
        dt_in(dt_in),
        dt_out(dt_out) {}
  bool operator==(const PackKey& other) const {
    return N == other.N && K == other.K && ld_in == other.ld_in &&
        ld_out == other.ld_out && dt_in == other.dt_in &&
        dt_out == other.dt_out;
  }
};

inline dnnl::memory::data_type get_dnnl_dtype(ScalarType dtype) {
  if (dtype == ScalarType::Float) {
    return dnnl::memory::data_type::f32;
  } else if (dtype == ScalarType::BFloat16) {
    return dnnl::memory::data_type::bf16;
  } else if (dtype == ScalarType::Half) {
    return dnnl::memory::data_type::f16;
  } else if (dtype == ScalarType::Byte) {
    return dnnl::memory::data_type::u8;
  } else if (dtype == ScalarType::Char) {
    return dnnl::memory::data_type::s8;
  } else {
    TORCH_CHECK(false, "get_dnnl_dtype expects float/bfloat16/half/int8 tensor input");
  }
}

template<typename key_t>
struct UnsafeUkernelKeyHasher {
  std::size_t operator()(const key_t& key) const;
};

template<>
std::size_t UnsafeUkernelKeyHasher<BrgemmKey>::operator()(const BrgemmKey& key) const {
  // Use M, N, K add_C, and ldc to compute hash to reduce the overhead as
  // batch size and data types are unlikely to change within the same kernel and
  // lda/ldb are likely to be related to M, K, N or use fixed values.
  std::size_t h = std::hash<int64_t>()(key.M);
  h = std::hash<int64_t>()(key.N) ^ (h << 1);
  h = std::hash<int64_t>()(key.K) ^ (h << 1);
  h = std::hash<bool>()(key.add_C) ^ (h << 1);
  h = std::hash<int64_t>()(key.ldc) ^ (h << 1);
  return h;
}

template<>
std::size_t UnsafeUkernelKeyHasher<PackKey>::operator()(const PackKey& key) const {
  // Use K and N to compute hash to reduce the overhead as
  // data types are unlikely to change and
  // ld_in/ld_out is likely to be related to K, N or use fixed values
  std::size_t h = std::hash<int64_t>()(key.K);
  h = std::hash<int64_t>()(key.N) ^ (h << 1);
  return h;
}

template <typename key_t, typename value_t>
struct KernelCache  {
  using kstore_t = std::unordered_map<key_t, std::shared_ptr<value_t>, UnsafeUkernelKeyHasher<key_t>>;
  static inline std::shared_ptr<value_t>&& fetch_or_create(
      const key_t& key,
      const std::function<std::shared_ptr<value_t>()>& callback) {
    auto&& search = get_store().find(key);
    if (search != get_store().end()) {
      return std::move(search->second);
    } else {
      get_store().insert({key, callback()});
      return std::move(get_store()[key]);
    }
  }

  static inline kstore_t& get_store() {
    static thread_local kstore_t cache_kernels;
    return cache_kernels;
  }
};

// Helper struct for convenient brgemm configuration
struct GemmHelper {
  GemmHelper(
      int64_t M,
      int64_t N,
      int64_t K,
      int64_t bs,
      int64_t ld_a,
      int64_t ld_b,
      int64_t ld_c,
      ScalarType dt_a,
      ScalarType dt_b,
      ScalarType dt_c,
      const bool add_C) {
    // Create brgemm
#if defined(ONEDNN_UKERNEL_1)
    brg = dnnl::ukernel::brgemm(
        M,
        N,
        K,
        bs,
        ld_a,
        ld_b,
        ld_c,
        get_dnnl_dtype(dt_a),
        get_dnnl_dtype(dt_b),
        get_dnnl_dtype(dt_c),
        1,
        add_C ? 1 : 0);
#elif defined(ONEDNN_UKERNEL_2)
    brg = dnnl::ukernel::brgemm(
        M,
        N,
        K,
        bs,
        ld_a,
        ld_b,
        ld_c,
        get_dnnl_dtype(dt_a),
        get_dnnl_dtype(dt_b),
        get_dnnl_dtype(dt_c));
    brg.set_add_C(add_C);
    brg.finalize();
#endif
    // Create a scratchpad buffer for the brgemm execution
    scratchpad = std::vector<uint8_t>(brg.get_scratchpad_size());
    // Prepare default vector of pairs of tensors A and B offsets for each batch.
    A_B_offsets.reserve(1);
    A_B_offsets[0] = std::make_pair(0, 0);
  }
  dnnl::ukernel::brgemm brg;
  std::vector<uint8_t> scratchpad;
  std::vector<std::pair<int64_t, int64_t>> A_B_offsets;
};

struct Brgemm : public KernelCache <BrgemmKey, GemmHelper> {
  // Fetch/create GemmHelper object and execute brgemm with batch size = 1
  template <typename scalar_t_a, typename scalar_t_b, typename scalar_t_c>
  static inline void call(
      int64_t M,
      int64_t N,
      int64_t K,
      int64_t ld_a,
      int64_t ld_b,
      int64_t ld_c,
      const bool add_C,
      const scalar_t_a* A,
      const scalar_t_b* B,
      scalar_t_c* C) {
    auto&& key = BrgemmKey(
        M,
        N,
        K,
        int64_t(1),
        ld_a,
        ld_b,
        ld_c,
        c10::CppTypeToScalarType<scalar_t_a>::value,
        c10::CppTypeToScalarType<scalar_t_b>::value,
        c10::CppTypeToScalarType<scalar_t_c>::value,
        add_C);
    // Fetch/create GemmHelper object
    auto&& value = fetch_or_create(key, [&]() {
      auto&& v = std::make_shared<GemmHelper>(
          M,
          N,
          K,
          1,
          ld_a,
          ld_b,
          ld_c,
          c10::CppTypeToScalarType<scalar_t_a>::value,
          c10::CppTypeToScalarType<scalar_t_b>::value,
          c10::CppTypeToScalarType<scalar_t_c>::value,
          add_C);
      (*v).brg.generate();
      return std::move(v);
    });
    if (get_current() != value) {
#if defined(ONEDNN_UKERNEL_1)
      dnnl::ukernel::brgemm::release_hw_context();
#endif
      ((*value).brg).set_hw_context();
      get_current() = value;
    }
    ((*value).brg)
        .execute(A, B, (*value).A_B_offsets, C, (*value).scratchpad.data());
  }

  static inline std::shared_ptr<GemmHelper>& get_current() {
    static thread_local std::shared_ptr<GemmHelper> current;
    return current;
  }

  static inline bool device_check(ScalarType dtype) {
    if (!at::globalContext().userEnabledMkldnn()) {
      return false;
    }
    if (dtype == ScalarType::Half) {
      static bool fp16_support = dnnl::get_effective_cpu_isa() >= dnnl::cpu_isa::avx512_core_fp16;
      return fp16_support;
    } else if (dtype == ScalarType::Float) {
      static bool fp32_support = dnnl::get_effective_cpu_isa() >= dnnl::cpu_isa::avx2;
      return fp32_support;
    } else if (dtype == ScalarType::BFloat16) {
      static bool bf16_support = dnnl::get_effective_cpu_isa() >= dnnl::cpu_isa::avx512_core;
      return bf16_support;
    }
    return false;
  }
};

#if defined(ONEDNN_UKERNEL_1)
using pack_t = dnnl::ukernel::brgemm_pack_B;
#elif defined(ONEDNN_UKERNEL_2)
using pack_t = dnnl::ukernel::transform;
#endif
struct Pack : public KernelCache <PackKey, pack_t> {
  static inline void call(
      int64_t K,
      int64_t N,
      int64_t ld_in,
      int64_t ld_out,
      ScalarType dt_in,
      ScalarType dt_out,
      const void* in,
      void* out) {
    auto&& key = PackKey(K, N, ld_in, ld_out, dt_in, dt_out);
    auto&& pack = fetch_or_create(key, [&]() {
      auto&& p = std::make_shared<pack_t>(
#if defined(ONEDNN_UKERNEL_1)
          K, N, ld_in, ld_out, get_dnnl_dtype(dt_in), get_dnnl_dtype(dt_out));
#elif defined(ONEDNN_UKERNEL_2)
          K, N, dnnl::ukernel::pack_type::no_trans, ld_in, ld_out, get_dnnl_dtype(dt_in), get_dnnl_dtype(dt_out));
#endif
      if (could_pack(dt_in)) {
        (*p).generate();
      }
      return std::move(p);
    });
    if (could_pack(dt_in)) {
      (*pack).execute(in, out);
    } else {
      TORCH_CHECK(false, "No need to pack");
    }
  }

  static inline bool could_pack(ScalarType dtype) {
    if (!at::globalContext().userEnabledMkldnn()) {
      return false;
    }
    if (dtype == ScalarType::Half) {
      static bool fp16_pack = dnnl::get_effective_cpu_isa() >= dnnl::cpu_isa::avx512_core_amx_fp16;
      return fp16_pack;
    } else if (dtype == ScalarType::BFloat16) {
      static bool bf16_pack = dnnl::get_effective_cpu_isa() >= dnnl::cpu_isa::avx512_core_amx;
      return bf16_pack;
    }
    return false;
  }
};
#endif

void brgemm(
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
    bool is_vnni) {

  TORCH_CHECK(!is_vnni,
    "Float Brgemm does not support vnni layout.");

#if defined(ONEDNN_UKERNEL_ENABLED)
  if (Brgemm::device_check(ScalarType::Float)) {
    Brgemm::call<float, float, float>(
      M, N, K, ld_a, ld_b, ld_c, add_C, A, B, C);
    return;
  }
#endif
  // fallback path
  auto beta = add_C ? 1 : 0;
  gemm(
    at::native::TransposeType::NoTranspose,
    at::native::TransposeType::NoTranspose,
    N, M, K, 1,
    B, ld_b, A, ld_a,
    beta, C, ld_c);
}

void brgemm(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t ld_a,
    int64_t ld_b,
    int64_t ld_c,
    const bool add_C,
    const at::BFloat16* A,
    const at::BFloat16* B,
    float* C,
    bool is_vnni) {
#if defined(ONEDNN_UKERNEL_ENABLED)
  if (is_vnni && Brgemm::device_check(ScalarType::BFloat16)) {
    Brgemm::call<at::BFloat16, at::BFloat16, float>(
      M, N, K, ld_a, ld_b, ld_c, add_C, A, B, C);
    return;
  }
#endif
  // fallback path
  TORCH_CHECK(!is_vnni,
    "BFloat16 Brgemm VNNI format is only supported on X64 when oneDNN ukernel is enabled and `amx` is supported");
  auto beta = add_C ? 1 : 0;
  gemm(
    at::native::TransposeType::NoTranspose,
    at::native::TransposeType::NoTranspose,
    N, M, K, 1,
    B, ld_b, A, ld_a,
    beta, C, ld_c);
}

void brgemm(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t ld_a,
    int64_t ld_b,
    int64_t ld_c,
    const bool add_C,
    const at::Half* A,
    const at::Half* B,
    float* C,
    bool is_vnni) {
#if defined(ONEDNN_UKERNEL_ENABLED)
  if (is_vnni && Brgemm::device_check(ScalarType::Half)) {
    Brgemm::call<at::Half, at::Half, float>(
      M, N, K, ld_a, ld_b, ld_c, add_C, A, B, C);
    return;
  }
#endif
  // fallback path
  TORCH_CHECK(!is_vnni,
    "Half Brgemm VNNI format is only supported on X64 when oneDNN ukernel is enabled and `amx_fp16` is supported");
  auto beta = add_C ? 1 : 0;
  gemm(
    at::native::TransposeType::NoTranspose,
    at::native::TransposeType::NoTranspose,
    N, M, K, 1,
    B, ld_b, A, ld_a,
    beta, C, ld_c);
}

void brgemm_release(bool is_vnni) {
#if defined(ONEDNN_UKERNEL_ENABLED)
  if (is_vnni) {
    dnnl::ukernel::brgemm::release_hw_context();
    Brgemm::get_current() = nullptr;
  }
#endif
}

void pack(
    int64_t K,
    int64_t N,
    int64_t ld_in,
    int64_t ld_out,
    ScalarType dt_in,
    ScalarType dt_out,
    const void* in,
    void* out) {
#if defined(ONEDNN_UKERNEL_ENABLED)
  Pack::call(K, N, ld_in, ld_out, dt_in, dt_out, in, out);
#else
  TORCH_CHECK(false, "pack is only supported on X64 with oneDNN ukernel enabled");
#endif
}

bool could_pack(ScalarType dt_in) {
#if defined(ONEDNN_UKERNEL_ENABLED)
  return Pack::could_pack(dt_in);
#else
  return false;
#endif
}

} // namespace at::native::cpublas
