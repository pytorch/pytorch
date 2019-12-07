#include <limits>
#include <algorithm>
#include <ATen/ATen.h>

#ifdef USE_BLAS
extern "C" void dscal_(int *n, double *a, double *x, int *incx);
extern "C" void sscal_(int *n, float *a, float *x, int *incx);
extern "C" void dgemv_(char *trans, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern "C" void sgemv_(char *trans, int *m, int *n, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy);
#endif

namespace at { namespace native {

namespace {

template <typename scalar_t>
constexpr inline bool scal_use_fast_path(int64_t n, int64_t incx) {
  return false;
}

template <typename scalar_t>
constexpr inline bool gemv_use_fast_path(int64_t m, int64_t n, int64_t lda, int64_t incx, int64_t incy) {
  return false;
}

template <typename scalar_t>
inline void scal_fast_path(int *n, scalar_t *a, scalar_t *x, int *incx) {
  TORCH_INTERNAL_ASSERT(false, "scal_fast_path shouldn't be called for this configuration");
}

template <typename scalar_t>
inline void gemv_fast_path(char *trans, int *m, int *n, scalar_t *alpha, scalar_t *a, int *lda, scalar_t *x, int *incx, scalar_t *beta, scalar_t *y, int *incy) {
  TORCH_INTERNAL_ASSERT(false, "gemv_fast_path shouldn't be called for this configuration");
}

#ifdef USE_BLAS
template <>
constexpr inline bool scal_use_fast_path<double>(int64_t n, int64_t incx) {
  auto intmax = std::numeric_limits<int>::max;
  return n <= intmax && incx <= intmax;
}

template <>
constexpr inline bool scal_use_fast_path<float>(int64_t n, int64_t incx) {
  return scal_use_fast_path<double>(n, incx);
}

template <>
inline void scal_fast_path<double>(int *n, double *a, double *x, int *incx) {
  dscal_(n, a, x, incx);
}

template <>
inline void scal_fast_path<float>(int *n, float *a, float *x, int *incx) {
  sscal_(n, a, x, incx);
}

template <>
constexpr inline bool gemv_use_fast_path<float>(int64_t m, int64_t n, int64_t lda, int64_t incx, int64_t incy) {
  auto intmax = std::numeric_limits<int>::max;
  return (m <= intmax) && (n <= intmax) && (lda <= intmax) &&
         (incx > 0) && (incx <= intmax) && (incy > 0) && (incy <= intmax);
}

template <>
constexpr inline bool gemv_use_fast_path<double>(int64_t m, int64_t n, int64_t lda, int64_t incx, int64_t incy) {
  return gemv_use_fast_path<float>(m, n, lda, incx, incy);
}

template <>
inline void gemv_fast_path<double>(char *trans, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy) {
  dgemv_(trans, m, n, alpha, a, lda, x, i_incx, beta, y, i_incy);
}

template <>
inline void gemv_fast_path<float>(char *trans, int *m, int *n, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy) {
  sgemv_(trans, m, n, alpha, a, lda, x, i_incx, beta, y, i_incy);
}
#endif

} // anonymous namespace

template <typename scalar_t>
void scal(int64_t n, scalar_t a, scalar_t *x, int64_t incx)
{
  if (n == 1) incx = 1;
  if (scal_use_fast_path<scalar_t>(n, incx)) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    scal_fast_path<scalar_t>(&i_n, &a, x, &i_incx);
    return;
  }
  for (int64_t i = 0; i < n; i++) {
    if (a == 0) {
      x[i * incx] = 0;
    } else {
      x[i * incx] *= a;
    }
  }
}

template<typename scalar_t>
bool gemv(char trans, int64_t m, int64_t n, scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy) {
  if(n == 1) lda = m;

  if (gemv_use_fast_path<scalar_t>(m, n, lda, incx, incy)) {
    TORCH_CHECK(lda >= std::max<int64_t>(1L, m), "lda should be at least max(1,", m, "), but have ", lda);
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    gemv_fast_path<scalar_t>(&trans, &i_m, &i_n, &alpha, a, &i_lda, x, &i_incx, &beta, y, &i_incy);
    return true;
  }

  if ((trans == 'T') || (trans == 't')) {
    for (int64_t i = 0; i < n; i++)
    {
      scalar_t sum = 0;
      scalar_t *row_ = a + lda * i;
      for (int64_t j = 0; j < m; j++) {
        sum += x[j * incx] * row_[j];
      }
      if (beta == 0) {
        y[i * incy] = alpha * sum;
      } else {
        y[i * incy] = beta * y[i * incy] + alpha * sum;
      }
    }
  } else {
    if (beta != 1) scal<scalar_t>(m, beta, y, incy);

    for (int64_t j = 0; j < n; j++) {
      scalar_t *column_ = a + lda * j;
      scalar_t z = alpha * x[j * incx];
      for (int64_t i = 0; i < m; i++) {
        y[i * incy] += z * column_[i];
      }
    }
  }
  return false;
}

#define DEFINE_GEMV(scalar_t, _) \
template bool gemv<scalar_t>(char trans, int64_t m, int64_t n, scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy);

AT_FORALL_SCALAR_TYPES_AND(BFloat16, DEFINE_GEMV);

}} // namespace at::native
