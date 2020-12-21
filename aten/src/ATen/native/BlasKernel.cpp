#include <limits>
#include <algorithm>
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <TH/THGeneral.h>

#if AT_BUILD_WITH_BLAS()
extern "C" double ddot_(int *n, double *x, int *incx, double *y, int *incy);
extern "C" void dscal_(int *n, double *a, double *x, int *incx);
extern "C" void sscal_(int *n, float *a, float *x, int *incx);
extern "C" void dgemv_(char *trans, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern "C" void sgemv_(char *trans, int *m, int *n, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy);

#ifdef BLAS_F2C
# define ffloat double
#else
# define ffloat float
#endif

#ifdef BLAS_USE_CBLAS_DOT
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
#endif // BLAS_USE_CBLAS_DOT
#endif // AT_BUILD_WITH_BLAS

namespace at { namespace native {

namespace blas_impl {

template <typename scalar_t>
bool scal_use_fast_path(int64_t n, int64_t incx) {
  return false;
}

template <typename scalar_t>
bool gemv_use_fast_path(int64_t m, int64_t n, int64_t lda, int64_t incx, int64_t incy) {
  return false;
}

template <typename scalar_t>
void scal_fast_path(int *n, scalar_t *a, scalar_t *x, int *incx) {
  TORCH_INTERNAL_ASSERT(false, "scal_fast_path shouldn't be called for this configuration");
}

template <typename scalar_t>
void gemv_fast_path(char *trans, int *m, int *n, scalar_t *alpha, scalar_t *a, int *lda, scalar_t *x, int *incx, scalar_t *beta, scalar_t *y, int *incy) {
  TORCH_INTERNAL_ASSERT(false, "gemv_fast_path shouldn't be called for this configuration");
}

#define INSTANTIATE(scalar_t)                                                                                                                                                     \
template bool scal_use_fast_path<scalar_t>(int64_t n, int64_t incx);                                                                                                              \
template bool gemv_use_fast_path<scalar_t>(int64_t m, int64_t n, int64_t lda, int64_t incx, int64_t incy);                                                                        \
template void gemv_fast_path<scalar_t>(char *trans, int *m, int *n, scalar_t *alpha, scalar_t *a, int *lda, scalar_t *x, int *incx, scalar_t *beta, scalar_t *y, int *incy);      \
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
void gemv_fast_path<double>(char *trans, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy) {
  dgemv_(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
void gemv_fast_path<float>(char *trans, int *m, int *n, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy) {
  sgemv_(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
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
  for (int64_t i = 0; i < n; i++) {
    if (a == scalar_t(0)) {
      x[i * incx] = 0;
    } else {
      x[i * incx] *= a;
    }
  }
}

template<typename scalar_t>
void gemv(char trans, int64_t m, int64_t n, scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy) {
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

  if ((trans == 'T') || (trans == 't')) {
    for (int64_t i = 0; i < n; i++)
    {
      scalar_t sum = 0;
      scalar_t *row_ = a + lda * i;
      for (int64_t j = 0; j < m; j++) {
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

    for (int64_t j = 0; j < n; j++) {
      scalar_t *column_ = a + lda * j;
      scalar_t z = alpha * x[j * incx];
      for (int64_t i = 0; i < m; i++) {
        //output values are ignored if beta is 0, and set to 0, nans and infs are not propagated
        if (j==0 && beta==scalar_t(0)) {
         y[i * incy] = scalar_t(0);
        }
        y[i * incy] += z * column_[i];
      }
    }
  }
  return;
}

#define INSTANTIATE(scalar_t, _) \
template void gemv<scalar_t>(char trans, int64_t m, int64_t n, scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy);
AT_FORALL_SCALAR_TYPES_AND(BFloat16, INSTANTIATE);
AT_FORALL_COMPLEX_TYPES(INSTANTIATE);
#undef INSTANTIATE

namespace blas_impl {
#if AT_BUILD_WITH_BLAS()
float dot_fast_path(int n, float* x, int incx, float* y, int incy) {
  return sdot_(&n, x, &incx, y, &incy);
}

double dot_fast_path(int n, double* x, int incx, double* y, int incy) {
  return ddot_(&n, x, &incx, y, &incy);
}

c10::complex<float> vdot_fast_path(int n, c10::complex<float>* x, int incx, c10::complex<float>* y, int incy) {
  c10::complex<float> result;
  cdotc_(reinterpret_cast<std::complex<float>* >(&result), &n, reinterpret_cast<std::complex<float>*>(x), &incx, reinterpret_cast<std::complex<float>*>(y), &incy);
  return result;
}

c10::complex<double> vdot_fast_path(int n, c10::complex<double>* x, int incx, c10::complex<double>* y, int incy) {
  c10::complex<double> result;
  zdotc_(reinterpret_cast<std::complex<double>* >(&result), &n, reinterpret_cast<std::complex<double>*>(x), &incx, reinterpret_cast<std::complex<double>*>(y), &incy);
  return result;
}

c10::complex<double> dot_fast_path(int n, c10::complex<double>* x, int incx, c10::complex<double>* y, int incy) {
  c10::complex<double> result;
  zdotu_(reinterpret_cast<std::complex<double>* >(&result), &n, reinterpret_cast<std::complex<double>*>(x), &incx, reinterpret_cast<std::complex<double>*>(y), &incy);
  return result;
}

c10::complex<float> dot_fast_path(int n, c10::complex<float>* x, int incx, c10::complex<float>* y, int incy) {
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
  int64_t i;
  scalar_t sum = 0;
  for (i = 0; i < n; i++) {
    sum += op(x[i * incx], y[i * incy]);
  }
  return sum;
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

}} // namespace at::native
