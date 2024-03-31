#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/Config.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <c10/util/complex.h>
#include <c10/util/irange.h>
#include <algorithm>
#include <climits>
#include <limits>

#if defined(__aarch64__) && !defined(C10_MOBILE)
#include <arm_neon.h>
#endif

namespace {

/// Wrapper for const_cast<T*> with type-inference.
///
/// Use this to call into APIs that are not const-correct.
template <typename T>
T* remove_const(const T* x) {
  return const_cast<T*>(x);
}

} // namespace

#if AT_BUILD_WITH_BLAS()
extern "C" double ddot_(int *n, double *x, int *incx, double *y, int *incy);
extern "C" void dscal_(int *n, double *a, double *x, int *incx);
extern "C" void sscal_(int *n, float *a, float *x, int *incx);
extern "C" void dgemv_(char *trans, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern "C" void sgemv_(char *trans, int *m, int *n, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy);

#if AT_BLAS_F2C()
# define ffloat double
#else
# define ffloat float
#endif

#if AT_BLAS_USE_CBLAS_DOT()
  extern "C" float cblas_sdot(const int n, const float *x, const int incx, const float *y, const int incy);
  extern "C" void cblas_cdotu_sub(const int n, const void *x, const int incx, const void *y, const int incy, void *dotu);
  extern "C" void cblas_zdotu_sub(const int n, const void *x, const int incx, const void *y, const int incy, void *dotu);
  extern "C" void cblas_cdotc_sub(const int n, const void *x, const int incx, const void *y, const int incy, void *dotc);
  extern "C" void cblas_zdotc_sub(const int n, const void *x, const int incx, const void *y, const int incy, void *dotc);

  static inline ffloat sdot_(const int *n, const float *x, const int *incx, const float *y, const int *incy)
  {
    return cblas_sdot(*n, x, *incx, y, *incy);
  }
  static inline void cdotu_(std::complex<float> *res, const int *n, const std::complex<float> *x, const int *incx,
  const std::complex<float> *y, const int *incy) {
    cblas_cdotu_sub(*n, x, *incx, y, *incy, res);
  }
  static inline void zdotu_(std::complex<double> *res, const int *n, const std::complex<double> *x, const int *incx,
  const std::complex<double> *y, const int *incy) {
    cblas_zdotu_sub(*n, x, *incx, y, *incy, res);
  }
  static inline void cdotc_(std::complex<float> *res, const int *n, const std::complex<float> *x, const int *incx,
  const std::complex<float> *y, const int *incy) {
    cblas_cdotc_sub(*n, x, *incx, y, *incy, res);
  }
  static inline void zdotc_(std::complex<double> *res, const int *n, const std::complex<double> *x, const int *incx,
  const std::complex<double> *y, const int *incy) {
    cblas_zdotc_sub(*n, x, *incx, y, *incy, res);
  }

#else
  extern "C" ffloat sdot_(int *n, float *x, int *incx, float *y, int *incy);
  extern "C" void cdotu_(std::complex<float> *res, int *n, std::complex<float> *x, int *incx, std::complex<float> *y, int *incy);
  extern "C" void zdotu_(std::complex<double> *res, int *n, std::complex<double> *x, int *incx, std::complex<double> *y, int *incy);
  extern "C" void cdotc_(std::complex<float> *res, int *n, std::complex<float> *x, int *incx, std::complex<float> *y, int *incy);
  extern "C" void zdotc_(std::complex<double> *res, int *n, std::complex<double> *x, int *incx, std::complex<double> *y, int *incy);
#endif // AT_BLAS_USE_CBLAS_DOT
#endif // AT_BUILD_WITH_BLAS

namespace at::native {

namespace blas_impl {
#if defined(__aarch64__) && !defined(C10_MOBILE)
void fp16_gemv_notrans(
    const int m,
    const int n,
    const float alpha,
    const float16_t* a,
    const int lda,
    const float16_t* x,
    const int incx,
    const float beta,
    float16_t* y,
    const int incy);

void fp16_gemv_trans(
    const int m,
    const int n,
    const float alpha,
    const float16_t* a,
    const int lda,
    const float16_t* x,
    const int incx,
    const float beta,
    float16_t* y,
    const int incy);
#endif

template <typename scalar_t>
bool scal_use_fast_path(C10_UNUSED int64_t n, C10_UNUSED int64_t incx) {
  return false;
}

template <typename scalar_t>
bool gemv_use_fast_path(C10_UNUSED int64_t m, C10_UNUSED int64_t n,
                        C10_UNUSED int64_t lda, C10_UNUSED int64_t incx, C10_UNUSED int64_t incy) {
  return false;
}

template <typename scalar_t>
void scal_fast_path(C10_UNUSED int *n, C10_UNUSED scalar_t *a, C10_UNUSED scalar_t *x, C10_UNUSED int *incx) {
  TORCH_INTERNAL_ASSERT(false, "scal_fast_path shouldn't be called for this configuration");
}

template <typename scalar_t>
void gemv_fast_path(C10_UNUSED const char *trans, C10_UNUSED const int *m, C10_UNUSED const int *n,
                    C10_UNUSED  const scalar_t *alpha, C10_UNUSED const scalar_t *a, C10_UNUSED const int *lda,
                    C10_UNUSED  const scalar_t *x, C10_UNUSED const int *incx, C10_UNUSED const scalar_t *beta,
                    C10_UNUSED  scalar_t *y, C10_UNUSED const int *incy) {
  TORCH_INTERNAL_ASSERT(false, "gemv_fast_path shouldn't be called for this configuration");
}

#define INSTANTIATE(scalar_t)                                                                                                                                                     \
template bool scal_use_fast_path<scalar_t>(int64_t n, int64_t incx);                                                                                                              \
template bool gemv_use_fast_path<scalar_t>(int64_t m, int64_t n, int64_t lda, int64_t incx, int64_t incy);                                                                        \
template void gemv_fast_path<scalar_t>(const char *trans, const int *m, const int *n, const scalar_t *alpha, const scalar_t *a, const int *lda, const scalar_t *x, const int *incx, const scalar_t *beta, scalar_t *y, const int *incy);      \
template void scal_fast_path<scalar_t>(int *n, scalar_t *a, scalar_t *x, int *incx);

#if AT_BUILD_WITH_BLAS()
template <>
bool scal_use_fast_path<double>(int64_t n, int64_t incx) {
  auto intmax = std::numeric_limits<int>::max();
  return n <= intmax && incx <= intmax;
}

template <>
bool scal_use_fast_path<float>(int64_t n, int64_t incx) {
  return scal_use_fast_path<double>(n, incx);
}

template <>
void scal_fast_path<double>(int *n, double *a, double *x, int *incx) {
  dscal_(n, a, x, incx);
}

template <>
void scal_fast_path<float>(int *n, float *a, float *x, int *incx) {
  sscal_(n, a, x, incx);
}

template <>
bool gemv_use_fast_path<float>(int64_t m, int64_t n, int64_t lda, int64_t incx, int64_t incy) {
  auto intmax = std::numeric_limits<int>::max();
  return (m <= intmax) && (n <= intmax) && (lda <= intmax) &&
         (incx > 0) && (incx <= intmax) && (incy > 0) && (incy <= intmax);
}

template <>
bool gemv_use_fast_path<double>(int64_t m, int64_t n, int64_t lda, int64_t incx, int64_t incy) {
  return gemv_use_fast_path<float>(m, n, lda, incx, incy);
}

template <>
void gemv_fast_path<double>(const char *trans, const int *m, const int *n, const double *alpha, const double *a, const int *lda, const double *x, const int *incx, const double *beta, double *y, const int *incy) {
  dgemv_(remove_const(trans), remove_const(m), remove_const(n), remove_const(alpha), remove_const(a), remove_const(lda), remove_const(x), remove_const(incx), remove_const(beta), y, remove_const(incy));
}

template <>
void gemv_fast_path<float>(const char *trans, const int *m, const int *n, const float *alpha, const float *a, const int *lda, const float *x, const int *incx, const float *beta, float *y, const int *incy) {
  sgemv_(remove_const(trans), remove_const(m), remove_const(n), remove_const(alpha), remove_const(a), remove_const(lda), remove_const(x), remove_const(incx), remove_const(beta), y, remove_const(incy));
}
#else
INSTANTIATE(float);
INSTANTIATE(double);
#endif // AT_BUILD_WITH_BLAS

INSTANTIATE(uint8_t);
INSTANTIATE(int8_t);
INSTANTIATE(int16_t);
INSTANTIATE(int);
INSTANTIATE(int64_t);
INSTANTIATE(c10::BFloat16);
#if defined(__aarch64__) && !defined(C10_MOBILE)
template <>
bool scal_use_fast_path<at::Half>(C10_UNUSED int64_t n, C10_UNUSED int64_t incx) {
  return false;
}

template <>
bool gemv_use_fast_path<at::Half>(
    C10_UNUSED int64_t m,
    C10_UNUSED int64_t n,
    C10_UNUSED int64_t lda,
    C10_UNUSED int64_t incx,
    C10_UNUSED int64_t incy) {
  return true;
}

#ifdef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
static inline float16_t reduce(float16x4_t x) {
        auto sum = vpadd_f16(x, x);
        return vget_lane_f16(vpadd_f16(sum, sum), 0);
}
static inline float16_t reduce(float16x8_t x) {
        return reduce(vadd_f16(vget_low_f16(x), vget_high_f16(x)));
}


static void fp16_gemv_trans_fp16_arith(const int m, const int n, const float16_t* a, const int lda, const float16_t *x, float16_t* y, int incy) {
  parallel_for(0, n / 4, 1, [&](int begin, int end) {
    for (auto i = begin * 4 ; i < end * 4; i += 4) {
      float16x8_t sum0Vec = vdupq_n_f16(0);
      float16x8_t sum1Vec = vdupq_n_f16(0);
      float16x8_t sum2Vec = vdupq_n_f16(0);
      float16x8_t sum3Vec = vdupq_n_f16(0);
      const auto row0 = a + lda * (i + 0);
      const auto row1 = a + lda * (i + 1);
      const auto row2 = a + lda * (i + 2);
      const auto row3 = a + lda * (i + 3);
      for (auto j = 0; j < m; j += 8) {
        float16x8_t xVec = vld1q_f16(x + j);
        float16x8_t a0Vec = vld1q_f16(row0 + j);
        sum0Vec = vaddq_f16(sum0Vec, vmulq_f16(a0Vec, xVec));
        float16x8_t a1Vec = vld1q_f16(row1 + j);
        sum1Vec = vaddq_f16(sum1Vec, vmulq_f16(a1Vec, xVec));
        float16x8_t a2Vec = vld1q_f16(row2 + j);
        sum2Vec = vaddq_f16(sum2Vec, vmulq_f16(a2Vec, xVec));
        float16x8_t a3Vec = vld1q_f16(row3 + j);
        sum3Vec = vaddq_f16(sum3Vec, vmulq_f16(a3Vec, xVec));
      }
      y[(i + 0) * incy] = reduce(sum0Vec);
      y[(i + 1) * incy] = reduce(sum1Vec);
      y[(i + 2) * incy] = reduce(sum2Vec);
      y[(i + 3) * incy] = reduce(sum3Vec);
    }
  });
}
#endif

static inline float reduce(float32x4_t x) {
        auto sum = vpaddq_f32(x, x);
        return vgetq_lane_f32(vpaddq_f32(sum, sum), 0);
}

static void fp16_gemv_trans_fp32_arith(const int m, const int n, const float16_t* a, const int lda, const float16_t *x, float16_t* y, int incy) {
  parallel_for(0, n / 4, 1, [&](int begin, int end) {
    for (auto i =  begin * 4 ; i < end * 4; i += 4) {
      float32x4_t sum0Vec = vdupq_n_f32(0);
      float32x4_t sum1Vec = vdupq_n_f32(0);
      float32x4_t sum2Vec = vdupq_n_f32(0);
      float32x4_t sum3Vec = vdupq_n_f32(0);
      const auto row0 = a + lda * (i + 0);
      const auto row1 = a + lda * (i + 1);
      const auto row2 = a + lda * (i + 2);
      const auto row3 = a + lda * (i + 3);
      for (auto j = 0; j < m; j += 4) {
        float32x4_t xVec = vcvt_f32_f16(vld1_f16(x + j));
        float32x4_t a0Vec = vcvt_f32_f16(vld1_f16(row0 + j));
        sum0Vec = vaddq_f32(sum0Vec, vmulq_f32(a0Vec, xVec));
        float32x4_t a1Vec = vcvt_f32_f16(vld1_f16(row1 + j));
        sum1Vec = vaddq_f32(sum1Vec, vmulq_f32(a1Vec, xVec));
        float32x4_t a2Vec = vcvt_f32_f16(vld1_f16(row2 + j));
        sum2Vec = vaddq_f32(sum2Vec, vmulq_f32(a2Vec, xVec));
        float32x4_t a3Vec = vcvt_f32_f16(vld1_f16(row3 + j));
        sum3Vec = vaddq_f32(sum3Vec, vmulq_f32(a3Vec, xVec));
      }
      y[(i + 0) * incy] = reduce(sum0Vec);
      y[(i + 1) * incy] = reduce(sum1Vec);
      y[(i + 2) * incy] = reduce(sum2Vec);
      y[(i + 3) * incy] = reduce(sum3Vec);
    }
  });
}

void fp16_gemv_trans(
    const int m,
    const int n,
    const float alpha,
    const float16_t* a,
    const int lda,
    const float16_t* x,
    const int incx,
    const float beta,
    float16_t* y,
    const int incy) {
  if (incx == 1 && alpha == 1.0 && beta == 0.0 && m % 4 == 0 && n % 4 == 0) {
#ifdef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
    return at::globalContext().allowFP16ReductionCPU() && m % 8 == 0 ? fp16_gemv_trans_fp16_arith(m, n, a, lda, x, y, incy)
                                                                     : fp16_gemv_trans_fp32_arith(m, n, a, lda, x, y, incy);
#else
    return fp16_gemv_trans_fp32_arith(m, n, a, lda, x, y, incy);
#endif
  }
  for (const auto i : c10::irange(n)) {
    float sum = 0;
    const auto row_ = a + lda * i;
    for (const auto j : c10::irange(m)) {
      sum += x[j * incx] * row_[j];
    }
    if (beta == 0.0) {
      y[i * incy] = alpha * sum;
    } else {
      y[i * incy] = beta * y[i * incy] + alpha * sum;
    }
  }
}


#ifdef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
static void fp16_gemv_notrans_fp16_arith(int m, int n, const float16_t* a, const int lda, const float16_t *x, float16_t *y) {
  for (auto j = 0; j < n; j++) {
    auto vecCol = vdup_n_f16(x[j]);
    const auto* column = a + lda * j;
    for (auto i = 0; i < m; i += 4) {
      auto yf16 = y + i;
      auto matRow = vld1_f16(column + i);
      auto resVec = j != 0 ? vld1_f16(yf16) : vdup_n_f16(0);
      resVec = vfma_lane_f16(resVec, matRow, vecCol, 0);
      vst1_f16(yf16, resVec);
    }
  }
}
#endif

static void fp16_gemv_notrans_fp32_arith(int m, int n, const float16_t* a, const int lda, const float16_t *x, float16_t *y) {
  std::vector<float> sum(m);
  for (auto j = 0; j < n; j++) {
    auto vecCol = vdup_n_f32(x[j]);
    const auto* column = a + lda * j;
    for (auto i = 0; i < m; i += 4) {
      auto sf32 = sum.data() + i;
      auto matRow = vcvt_f32_f16(vld1_f16(column + i));
      auto resVec = j != 0 ? vld1q_f32(sf32) : vdupq_n_f32(0);
      resVec = vfmaq_lane_f32(resVec, matRow, vecCol, 0);
      vst1q_f32(sf32, resVec);
    }
  }

  for (auto i = 0; i < m; i+= 4) {
    vst1_f16(y + i, vcvt_f16_f32(vld1q_f32(sum.data() + i)));
  }
}

void fp16_gemv_notrans(
    const int m,
    const int n,
    const float alpha,
    const float16_t* a,
    const int lda,
    const float16_t* x,
    const int incx,
    const float beta,
    float16_t* y,
    const int incy) {
  if (incx == 1 && alpha == 1.0 && beta == 0.0 && m % 4 == 0 && incy == 1) {
#ifdef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
    return at::globalContext().allowFP16ReductionCPU() ? fp16_gemv_notrans_fp16_arith(m, n, a, lda, x, y)
                                                       : fp16_gemv_notrans_fp32_arith(m, n, a, lda, x, y);
#else
    return fp16_gemv_notrans_fp32_arith(m, n, a, lda, x, y);
#endif
  }
  std::vector<float> sum(m);
  for (const auto j : c10::irange(n)) {
    const auto* column_ = a + lda * j;
    auto z = alpha * x[j * incx];
    for (const auto i : c10::irange(m)) {
      sum[i] += z * column_[i];
    }
  }
  if (beta == 0.0) {
    for (const auto i : c10::irange(m)) {
      y[i * incy] = sum[i];
    }
  } else {
    for (const auto i : c10::irange(m)) {
      y[i * incy] += sum[i];
    }
  }
}

template <>
void gemv_fast_path<at::Half>(
    const char* trans,
    const int* m,
    const int* n,
    const at::Half* alpha,
    const at::Half* a,
    const int* lda,
    const at::Half* x,
    const int* incx,
    const at::Half* beta,
    at::Half* y,
    const int* incy) {
  using namespace c10::detail;
  if ((trans[0] == 'T') || (trans[0] == 't')) {
    fp16_gemv_trans(
        *m,
        *n,
        fp16_from_bits(alpha->x),
        reinterpret_cast<const float16_t*>(a),
        *lda,
        reinterpret_cast<const float16_t*>(x),
        *incx,
        fp16_from_bits(beta->x),
        reinterpret_cast<float16_t*>(y),
        *incy);
  } else {
    fp16_gemv_notrans(
        *m,
        *n,
        fp16_from_bits(alpha->x),
        reinterpret_cast<const float16_t*>(a),
        *lda,
        reinterpret_cast<const float16_t*>(x),
        *incx,
        fp16_from_bits(beta->x),
        reinterpret_cast<float16_t*>(y),
        *incy);
  }
}
#else
INSTANTIATE(c10::Half);
#endif
#undef INSTANTIATE

} // namespace blas_impl

template <typename scalar_t>
inline void scal(int64_t n, scalar_t a, scalar_t *x, int64_t incx)
{
  if (n == 1) incx = 1;
  if (blas_impl::scal_use_fast_path<scalar_t>(n, incx)) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    blas_impl::scal_fast_path<scalar_t>(&i_n, &a, x, &i_incx);
    return;
  }
  for (const auto i : c10::irange(n)) {
    if (a == scalar_t(0)) {
      x[i * incx] = 0;
    } else {
      x[i * incx] *= a;
    }
  }
}

template<typename scalar_t>
void gemv(char trans, int64_t m, int64_t n, scalar_t alpha, const scalar_t *a, int64_t lda, const scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy) {
  if(n == 1) lda = m;

  if (blas_impl::gemv_use_fast_path<scalar_t>(m, n, lda, incx, incy)) {
    TORCH_CHECK(lda >= std::max<int64_t>(1L, m), "lda should be at least max(1,", m, "), but have ", lda);
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    blas_impl::gemv_fast_path<scalar_t>(&trans, &i_m, &i_n, &alpha, a, &i_lda, x, &i_incx, &beta, y, &i_incy);
    return;
  }

  using opmath_t = at::opmath_type<scalar_t>;
  if ((trans == 'T') || (trans == 't')) {
    for (const auto i : c10::irange(n)) {
      opmath_t sum = 0;
      const scalar_t *row_ = a + lda * i;
      for (const auto j : c10::irange(m)) {
        sum += x[j * incx] * row_[j];
      }
      if (beta == scalar_t(0)) {
        y[i * incy] = alpha * sum;
      } else {
        y[i * incy] = beta * y[i * incy] + alpha * sum;
      }
    }
  } else {
    if (beta != scalar_t(1) && beta != scalar_t(0)) scal<scalar_t>(m, beta, y, incy);

    constexpr bool is_low_precision = !std::is_same_v<opmath_t, scalar_t>;
    std::vector<opmath_t> sum;
    if constexpr (is_low_precision) {
      sum.resize(m);
    }
    for (const auto j : c10::irange(n)) {
      const scalar_t *column_ = a + lda * j;
      opmath_t z = alpha * static_cast<opmath_t>(x[j * incx]);
      for (const auto i : c10::irange(m)) {
        //output values are ignored if beta is 0, and set to 0, nans and infs are not propagated
        if (j==0 && beta==scalar_t(0)) {
          if constexpr (!is_low_precision) {
            y[i * incy] = 0;
          }
        }
        if constexpr (is_low_precision) {
          sum[i] += z * column_[i];
        } else {
          y[i * incy] += z * column_[i];
        }
      }
    }
    if constexpr (is_low_precision) {
      if (beta == scalar_t(0)) {
        for (const auto i : c10::irange(m)) {
          y[i * incy] = sum[i];
        }
      } else {
        for (const auto i : c10::irange(m)) {
          y[i * incy] += sum[i];
        }
      }
    }
  }
  return;
}

#define INSTANTIATE(scalar_t, _) \
template void gemv<scalar_t>(char trans, int64_t m, int64_t n, scalar_t alpha, const scalar_t *a, int64_t lda, const scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy);
AT_FORALL_SCALAR_TYPES_AND2(BFloat16, Half, INSTANTIATE);
AT_FORALL_COMPLEX_TYPES(INSTANTIATE);
#undef INSTANTIATE

namespace blas_impl {
#if AT_BUILD_WITH_BLAS()
static float dot_fast_path(int n, float* x, int incx, float* y, int incy) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return sdot_(&n, x, &incx, y, &incy);
}

static double dot_fast_path(int n, double* x, int incx, double* y, int incy) {
  return ddot_(&n, x, &incx, y, &incy);
}

static c10::complex<float> vdot_fast_path(int n, c10::complex<float>* x, int incx, c10::complex<float>* y, int incy) {
  c10::complex<float> result;
  cdotc_(reinterpret_cast<std::complex<float>* >(&result), &n, reinterpret_cast<std::complex<float>*>(x), &incx, reinterpret_cast<std::complex<float>*>(y), &incy);
  return result;
}

static c10::complex<double> vdot_fast_path(int n, c10::complex<double>* x, int incx, c10::complex<double>* y, int incy) {
  c10::complex<double> result;
  zdotc_(reinterpret_cast<std::complex<double>* >(&result), &n, reinterpret_cast<std::complex<double>*>(x), &incx, reinterpret_cast<std::complex<double>*>(y), &incy);
  return result;
}

static c10::complex<double> dot_fast_path(int n, c10::complex<double>* x, int incx, c10::complex<double>* y, int incy) {
  c10::complex<double> result;
  zdotu_(reinterpret_cast<std::complex<double>* >(&result), &n, reinterpret_cast<std::complex<double>*>(x), &incx, reinterpret_cast<std::complex<double>*>(y), &incy);
  return result;
}

static c10::complex<float> dot_fast_path(int n, c10::complex<float>* x, int incx, c10::complex<float>* y, int incy) {
  c10::complex<float> result;
  cdotu_(reinterpret_cast<std::complex<float>* >(&result), &n, reinterpret_cast<std::complex<float>*>(x), &incx, reinterpret_cast<std::complex<float>*>(y), &incy);
  return result;
}
#endif

template <typename scalar_t, typename Functor>
scalar_t dot_naive(
    int64_t n,
    scalar_t* x,
    int64_t incx,
    scalar_t* y,
    int64_t incy,
    Functor op) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t i;
  using opmath_t = at::opmath_type<scalar_t>;
  opmath_t sum = 0;
  for (i = 0; i < n; i++) {
    sum += op(static_cast<opmath_t>(x[i * incx]), static_cast<opmath_t>(y[i * incy]));
  }
  return static_cast<scalar_t>(sum);
}

} // namespace blas_impl

template <typename scalar_t>
scalar_t dot_impl_floating(int64_t n, scalar_t* x, int64_t incx, scalar_t* y, int64_t incy)
{
  if (n == 1) {
    incx = 1;
    incy = 1;
  }
#if AT_BUILD_WITH_BLAS()
        if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
          return blas_impl::dot_fast_path(n, x, incx, y, incy);
        } else {
          return blas_impl::dot_naive(n, x, incx, y, incy, std::multiplies<scalar_t>{});
        }
#else
        { return blas_impl::dot_naive(n, x, incx, y, incy, std::multiplies<scalar_t>{}); }
#endif
}

template <typename scalar_t>
scalar_t dot_impl(int64_t n, scalar_t* x, int64_t incx, scalar_t* y, int64_t incy) {
  if (n == 1) {
    incx = 1;
    incy = 1;
  }
  return blas_impl::dot_naive(n, x, incx, y, incy, std::multiplies<scalar_t>{});
}

template <>
float dot_impl(int64_t n, float* x, int64_t incx, float* y, int64_t incy) {
  return dot_impl_floating(n, x, incx, y, incy);
}

template <>
double dot_impl(int64_t n, double* x, int64_t incx, double* y, int64_t incy) {
  return dot_impl_floating(n, x, incx, y, incy);
}

template <>
c10::complex<double> dot_impl(int64_t n, c10::complex<double>* x, int64_t incx, c10::complex<double>* y, int64_t incy) {
  return dot_impl_floating(n, x, incx, y, incy);
}

template <>
c10::complex<float> dot_impl(int64_t n, c10::complex<float>* x, int64_t incx, c10::complex<float>* y, int64_t incy) {
  return dot_impl_floating(n, x, incx, y, incy);
}

namespace {
template <typename scalar_t>
struct vdot_op {
  scalar_t operator()(scalar_t x, scalar_t y) {
    return std::conj(x) * y;
  }
};
} // anonymous namespace

template <typename scalar_t>
scalar_t vdot_impl(int64_t n, scalar_t* x, int64_t incx, scalar_t* y, int64_t incy) {
  if (n == 1) {
    incx = 1;
    incy = 1;
  }
#if AT_BUILD_WITH_BLAS()
        if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
          return blas_impl::vdot_fast_path(n, x, incx, y, incy);
        } else {
          return blas_impl::dot_naive(n, x, incx, y, incy, vdot_op<scalar_t>{});
        }
#else
        { return blas_impl::dot_naive(n, x, incx, y, incy, vdot_op<scalar_t>{}); }
#endif
}

// Skip reinstantiating the explicitly specialized types `float` and `double`.
#define INSTANTIATE_DOT_IMPL(scalar_t)  \
  template scalar_t dot_impl<scalar_t>( \
      int64_t n, scalar_t * x, int64_t incx, scalar_t * y, int64_t incy);
INSTANTIATE_DOT_IMPL(uint8_t);
INSTANTIATE_DOT_IMPL(int8_t);
INSTANTIATE_DOT_IMPL(int16_t);
INSTANTIATE_DOT_IMPL(int);
INSTANTIATE_DOT_IMPL(int64_t);
INSTANTIATE_DOT_IMPL(c10::Half);
INSTANTIATE_DOT_IMPL(c10::BFloat16);

#define INSTANTIATE_VDOT_IMPL(scalar_t)  \
  template scalar_t vdot_impl<scalar_t>( \
      int64_t n, scalar_t * x, int64_t incx, scalar_t * y, int64_t incy);
INSTANTIATE_VDOT_IMPL(c10::complex<float>);
INSTANTIATE_VDOT_IMPL(c10::complex<double>);

#undef INSTANTIATE_DOT_IMPL

} // namespace at::native
