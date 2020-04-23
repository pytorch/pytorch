#include <limits>
#include <algorithm>
#include <ATen/ATen.h>
#include <ATen/Config.h>

#if AT_BUILD_WITH_BLAS()
extern "C" void dscal_(int *n, double *a, double *x, int *incx);
extern "C" void sscal_(int *n, float *a, float *x, int *incx);
extern "C" void dgemv_(char *trans, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern "C" void sgemv_(char *trans, int *m, int *n, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy);
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

}}} // namespace at::native::blas_impl
