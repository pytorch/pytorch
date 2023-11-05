#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/mkl/LinearAlgebra.h>
#include <ATen/Config.h>

#if !AT_MKL_ENABLED()

namespace at { namespace native {

void mkl_gemm_batched(
    const TransposeType trans_A, const TransposeType trans_B,
    const int batch_size, const int M, const int N, const int K, const float alpha,
    const float** A, const int lda, const float** B, const int ldb, const float beta,
    float** C, const int ldc) {
  TORCH_INTERNAL_ASSERT(false, "mkl_gemm_batched: ATen not compiled with MKL support");
}

void mkl_gemm_batched(
    const TransposeType trans_A, const TransposeType trans_B,
    const int batch_size, const int M, const int N, const int K, const double alpha,
    const double** A, const int lda, const double** B, const int ldb, const double beta,
    double** C, const int ldc) {
  TORCH_INTERNAL_ASSERT(false, "mkl_gemm_batched: ATen not compiled with MKL support");
}

void mkl_gemm_batched(
    const TransposeType trans_A, const TransposeType trans_B,
    const int batch_size, const int M, const int N, const int K, const c10::complex<float> alpha,
    const c10::complex<float>** A, const int lda, const c10::complex<float>** B, const int ldb,
    const c10::complex<float> beta, c10::complex<float>** C, const int ldc) {
  TORCH_INTERNAL_ASSERT(false, "mkl_gemm_batched: ATen not compiled with MKL support");
}

void mkl_gemm_batched(
    const TransposeType trans_A, const TransposeType trans_B,
    const int batch_size, const int M, const int N, const int K, const c10::complex<double> alpha,
    const c10::complex<double>** A, const int lda, const c10::complex<double>** B, const int ldb,
    const c10::complex<double> beta, c10::complex<double>** C, const int ldc) {
  TORCH_INTERNAL_ASSERT(false, "mkl_gemm_batched: ATen not compiled with MKL support");
}

}}

#else // AT_MKL_ENABLED

#include <mkl.h>
#include <c10/util/irange.h>

namespace at { namespace native {

static CBLAS_TRANSPOSE to_cblas(TransposeType x) {
  switch (x) {
    case TransposeType::NoTranspose: return CblasNoTrans;
    case TransposeType::Transpose: return CblasTrans;
    case TransposeType::ConjTranspose: return CblasConjTrans;
  }
  TORCH_INTERNAL_ASSERT(false, "Unknown TransposeType");
}

void mkl_gemm_batched(
    const TransposeType trans_A, const TransposeType trans_B,
    const int batch_size_, const int M, const int N, const int K, const float alpha,
    const float** A, const int lda_, const float** B, const int ldb_, const float beta,
    float** C, const int ldc_) {
  auto transa_cblas = to_cblas(trans_A);
  auto transb_cblas = to_cblas(trans_B);
  MKL_INT batch_size = batch_size_;
  MKL_INT m = M;
  MKL_INT n = N;
  MKL_INT k = K;
  MKL_INT lda = lda_;
  MKL_INT ldb = ldb_;
  MKL_INT ldc = ldc_;
  cblas_sgemm_batch(CblasColMajor, &transa_cblas, &transb_cblas, &m, &n, &k, &alpha,
                    A, &lda, B, &ldb, &beta, C, &ldc, 1, &batch_size);
}

void mkl_gemm_batched(
    const TransposeType trans_A, const TransposeType trans_B,
    const int batch_size_, const int M, const int N, const int K, const double alpha,
    const double** A, const int lda_, const double** B, const int ldb_, const double beta,
    double** C, const int ldc_) {
  auto transa_cblas = to_cblas(trans_A);
  auto transb_cblas = to_cblas(trans_B);
  MKL_INT batch_size = batch_size_;
  MKL_INT m = M;
  MKL_INT n = N;
  MKL_INT k = K;
  MKL_INT lda = lda_;
  MKL_INT ldb = ldb_;
  MKL_INT ldc = ldc_;
  cblas_dgemm_batch(CblasColMajor, &transa_cblas, &transb_cblas, &m, &n, &k, &alpha,
                    A, &lda, B, &ldb, &beta, C, &ldc, 1, &batch_size);
}

void mkl_gemm_batched(
    const TransposeType trans_A, const TransposeType trans_B,
    const int batch_size_, const int M, const int N, const int K, const c10::complex<float> alpha,
    const c10::complex<float>** A, const int lda_, const c10::complex<float>** B, const int ldb_,
    const c10::complex<float> beta, c10::complex<float>** C, const int ldc_) {
  auto transa_cblas = to_cblas(trans_A);
  auto transb_cblas = to_cblas(trans_B);
  MKL_INT batch_size = batch_size_;
  MKL_INT m = M;
  MKL_INT n = N;
  MKL_INT k = K;
  MKL_INT lda = lda_;
  MKL_INT ldb = ldb_;
  MKL_INT ldc = ldc_;
  cblas_cgemm_batch(CblasColMajor, &transa_cblas, &transb_cblas, &m, &n, &k,
                    reinterpret_cast<const void*>(&alpha),
                    reinterpret_cast<const void**>(A), &lda, reinterpret_cast<const void**>(B), &ldb,
                    reinterpret_cast<const void*>(&beta), reinterpret_cast<void**>(C), &ldc, 1, &batch_size);
}

void mkl_gemm_batched(
    const TransposeType trans_A, const TransposeType trans_B,
    const int batch_size_, const int M, const int N, const int K, const c10::complex<double> alpha,
    const c10::complex<double>** A, const int lda_, const c10::complex<double>** B, const int ldb_,
    const c10::complex<double> beta, c10::complex<double>** C, const int ldc_) {
  auto transa_cblas = to_cblas(trans_A);
  auto transb_cblas = to_cblas(trans_B);
  MKL_INT batch_size = batch_size_;
  MKL_INT m = M;
  MKL_INT n = N;
  MKL_INT k = K;
  MKL_INT lda = lda_;
  MKL_INT ldb = ldb_;
  MKL_INT ldc = ldc_;
  cblas_zgemm_batch(CblasColMajor, &transa_cblas, &transb_cblas, &m, &n, &k,
                    reinterpret_cast<const void*>(&alpha),
                    reinterpret_cast<const void**>(A), &lda, reinterpret_cast<const void**>(B), &ldb,
                    reinterpret_cast<const void*>(&beta), reinterpret_cast<void**>(C), &ldc, 1, &batch_size);
}

}} // namespace at::native

#endif
