#pragma once

#include <ATen/OpMathType.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TransposeType.h>
#include <c10/util/complex.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Scalar.h>


namespace at::native::cpublas {

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
    const Scalar& alpha,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const Scalar& beta,
    void *c, int64_t ldc);

DECLARE_DISPATCH(gemm_fn, gemm_stub)

template <typename scalar_t>
void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    at::opmath_type<scalar_t> alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    at::opmath_type<scalar_t> beta,
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
    float alpha,
    const at::BFloat16 *a, int64_t lda,
    const at::BFloat16 *b, int64_t ldb,
    float beta,
    at::BFloat16 *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const float alpha,
    const at::BFloat16 *a, int64_t lda,
    const at::BFloat16 *b, int64_t ldb,
    const float beta,
    float *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const at::Half *a, int64_t lda,
    const at::Half *b, int64_t ldb,
    float beta,
    at::Half *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const float alpha,
    const at::Half *a, int64_t lda,
    const at::Half *b, int64_t ldb,
    const float beta,
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

template <typename scalar_t>
void gemm_batched(
    TransposeType transa, TransposeType transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t * const *a, int64_t lda,
    const scalar_t * const *b, int64_t ldb,
    const scalar_t beta,
    scalar_t * const *c, int64_t ldc);

template <typename scalar_t>
void gemm_batched_with_stride(
    TransposeType transa, TransposeType transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t *a, int64_t lda, int64_t batch_stride_a,
    const scalar_t *b, int64_t ldb, int64_t batch_stride_b,
    scalar_t beta,
    scalar_t *c, int64_t ldc, int64_t batch_stride_c);

using axpy_fn = void(*)(at::ScalarType type, int64_t n, const Scalar& a, const void *x, int64_t incx, void *y, int64_t incy);

DECLARE_DISPATCH(axpy_fn, axpy_stub)

template<typename scalar_t>
void axpy(int64_t n, scalar_t a, const scalar_t *x, int64_t incx, scalar_t *y, int64_t incy){
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  axpy_stub(
      kCPU, c10::CppTypeToScalarType<scalar_t>::value,
      n, a, x, incx, y, incy);
}

void axpy(int64_t n, double a, const double *x, int64_t incx, double *y, int64_t incy);
void axpy(int64_t n, float a, const float *x, int64_t incx, float *y, int64_t incy);
void axpy(int64_t n, c10::complex<double> a, const c10::complex<double> *x, int64_t incx, c10::complex<double> *y, int64_t incy);
void axpy(int64_t n, c10::complex<float> a, const c10::complex<float> *x, int64_t incx, c10::complex<float> *y, int64_t incy);

using copy_fn = void(*)(at::ScalarType type, int64_t n, const void *x, int64_t incx, void *y, int64_t incy);

DECLARE_DISPATCH(copy_fn, copy_stub)

template<typename scalar_t>
void copy(int64_t n, const scalar_t *x, int64_t incx, scalar_t *y, int64_t incy) {
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }
  copy_stub(
      kCPU, c10::CppTypeToScalarType<scalar_t>::value,
      n, x, incx, y, incy);
}

void copy(int64_t n, const double *x, int64_t incx, double *y, int64_t incy);
void copy(int64_t n, const float *x, int64_t incx, float *y, int64_t incy);
void copy(int64_t n, const c10::complex<double> *x, int64_t incx, c10::complex<double> *y, int64_t incy);
void copy(int64_t n, const c10::complex<float> *x, int64_t incx, c10::complex<float> *y, int64_t incy);

// Batch-reduce GEMM
// Operates by the following formula:
// C = SUM(A[i] x B[i]) + C if add_C is true, i = 0 to batch size
// A Base pointer to a tensor A.
// B Base pointer to a tensor B.
// C Pointer to a tensor C (accumulation buffer).
// Note only batch size 1 is used currently
TORCH_API void brgemm(
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
    bool is_vnni = true);

TORCH_API void brgemm(
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
    bool is_vnni = true);

TORCH_API void brgemm(
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
    bool is_vnni = false);

// Release brgemm hardware context
TORCH_API void brgemm_release(bool is_vnni = true);

// Pack B matrix to get better performance if needed
void pack(
    int64_t K,
    int64_t N,
    int64_t ld_in,
    int64_t ld_out,
    ScalarType dt_in,
    ScalarType dt_out,
    const void* in,
    void* out);

// Whether pack is supported in the platform.
TORCH_API bool could_pack(ScalarType dt_in);

} // namespace at::native::cpublas
