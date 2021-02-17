#pragma once

#include <ATen/native/DispatchStub.h>
#include <c10/util/complex.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Scalar.h>

namespace at {
namespace native {
namespace cpublas {

enum TransposeType {
  Transpose,
  NoTranspose,
  // ConjTranspose, -- Not implemented
};

namespace internal {
void normalize_last_dims(
  TransposeType transa, TransposeType transb,
  int64_t m, int64_t n, int64_t k,
  int64_t *lda, int64_t *ldb, int64_t *ldc);
}  // namespace internal

using gemm_fn = void(*)(
    at::ScalarType type,
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    Scalar alpha,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    Scalar beta,
    void *c, int64_t ldc);

DECLARE_DISPATCH(gemm_fn, gemm_stub);

template <typename scalar_t>
void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    scalar_t beta,
    scalar_t *c, int64_t ldc) {
  internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
  gemm_stub(
    kCPU, c10::CppTypeToScalarType<scalar_t>::value,
    transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    double alpha,
    const double *a, int64_t lda,
    const double *b, int64_t ldb,
    double beta,
    double *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const float *a, int64_t lda,
    const float *b, int64_t ldb,
    float beta,
    float *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    c10::complex<double> alpha,
    const c10::complex<double> *a, int64_t lda,
    const c10::complex<double> *b, int64_t ldb,
    c10::complex<double> beta,
    c10::complex<double> *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    c10::complex<float> alpha,
    const c10::complex<float> *a, int64_t lda,
    const c10::complex<float> *b, int64_t ldb,
    c10::complex<float> beta,
    c10::complex<float> *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    int64_t alpha,
    const int64_t *a, int64_t lda,
    const int64_t *b, int64_t ldb,
    int64_t beta,
    int64_t *c, int64_t ldc);

template<typename T>
void THBlas_axpy(int64_t n, T a, T *x, int64_t incx, T *y, int64_t incy){
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  int64_t i;
  for(i = 0; i < n; i++)
    y[i*incy] += a*x[i*incx];
}

void THBlas_axpy(int64_t n, double a, double *x, int64_t incx, double *y, int64_t incy);
void THBlas_axpy(int64_t n, float a, float *x, int64_t incx, float *y, int64_t incy);
void THBlas_axpy(int64_t n, c10::complex<double> a, c10::complex<double> *x, int64_t incx, c10::complex<double> *y, int64_t incy);
void THBlas_axpy(int64_t n, c10::complex<float> a, c10::complex<float> *x, int64_t incx, c10::complex<float> *y, int64_t incy);

template<typename T>
void THBlas_copy(int64_t n, T *x, int64_t incx, T *y, int64_t incy) {
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  int64_t i;
  for(i = 0; i < n; i++)
    y[i*incy] = x[i*incx];
}

void THBlas_copy(int64_t n, double *x, int64_t incx, double *y, int64_t incy);
void THBlas_copy(int64_t n, float *x, int64_t incx, float *y, int64_t incy);
void THBlas_copy(int64_t n, c10::complex<double> *x, int64_t incx, c10::complex<double> *y, int64_t incy);
void THBlas_copy(int64_t n, c10::complex<float> *x, int64_t incx, c10::complex<float> *y, int64_t incy);

}}}  // namespace at::native::cpublas
