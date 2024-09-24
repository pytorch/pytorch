#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/mkl/LinearAlgebra.h>
#include <ATen/Config.h>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-parameter")
#if !AT_MKL_ENABLED()

namespace at { namespace native {

void mkl_gemm_batched(
    const TransposeType trans_A, const TransposeType trans_B,
    const MKL_INT batch_size, const MKL_INT M, const MKL_INT N, const MKL_INT K, const float alpha,
    const float** A, const MKL_INT lda, const float** B, const MKL_INT ldb, const float beta,
    float** C, const MKL_INT ldc) {
  TORCH_INTERNAL_ASSERT(false, "mkl_gemm_batched: ATen not compiled with MKL support");
}

void mkl_gemm_batched(
    const TransposeType trans_A, const TransposeType trans_B,
    const MKL_INT batch_size, const MKL_INT M, const MKL_INT N, const MKL_INT K, const double alpha,
    const double** A, const MKL_INT lda, const double** B, const MKL_INT ldb, const double beta,
    double** C, const MKL_INT ldc) {
  TORCH_INTERNAL_ASSERT(false, "mkl_gemm_batched: ATen not compiled with MKL support");
}

void mkl_gemm_batched(
    const TransposeType trans_A, const TransposeType trans_B,
    const MKL_INT batch_size, const MKL_INT M, const MKL_INT N, const MKL_INT K, const c10::complex<float> alpha,
    const c10::complex<float>** A, const MKL_INT lda, const c10::complex<float>** B, const MKL_INT ldb,
    const c10::complex<float> beta, c10::complex<float>** C, const MKL_INT ldc) {
  TORCH_INTERNAL_ASSERT(false, "mkl_gemm_batched: ATen not compiled with MKL support");
}

void mkl_gemm_batched(
    const TransposeType trans_A, const TransposeType trans_B,
    const MKL_INT batch_size, const MKL_INT M, const MKL_INT N, const MKL_INT K, const c10::complex<double> alpha,
    const c10::complex<double>** A, const MKL_INT lda, const c10::complex<double>** B, const MKL_INT ldb,
    const c10::complex<double> beta, c10::complex<double>** C, const MKL_INT ldc) {
  TORCH_INTERNAL_ASSERT(false, "mkl_gemm_batched: ATen not compiled with MKL support");
}

void mkl_gemm_bf16bf16f32(
    TransposeType trans_A, TransposeType trans_B,
    MKL_INT M, MKL_INT N, MKL_INT K, const float alpha,
    const c10::BFloat16* A, MKL_INT lda, const c10::BFloat16* B, MKL_INT ldb,
    const float beta, float* C, MKL_INT ldc) {
  TORCH_INTERNAL_ASSERT(false, "mkl_gemm_bf16bf16f32: ATen not compiled with MKL support");
}

void mkl_gemm_f16f16f32(
    TransposeType trans_A, TransposeType trans_B,
    int M, int N, int K, const float alpha,
    const c10::Half* A, int lda, const c10::Half* B, int ldb,
    const float beta, float* C, int ldc) {
  TORCH_INTERNAL_ASSERT(false, "mkl_gemm_f16f16f32: ATen not compiled with MKL support");
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
    const MKL_INT batch_size, const MKL_INT M, const MKL_INT N, const MKL_INT K, const float alpha,
    const float** A, const MKL_INT lda, const float** B, const MKL_INT ldb, const float beta,
    float** C, const MKL_INT ldc) {
  auto transa_cblas = to_cblas(trans_A);
  auto transb_cblas = to_cblas(trans_B);
  cblas_sgemm_batch(CblasColMajor, &transa_cblas, &transb_cblas, &M, &N, &K, &alpha,
                    A, &lda, B, &ldb, &beta, C, &ldc, 1, &batch_size);
}

void mkl_gemm_batched(
    const TransposeType trans_A, const TransposeType trans_B,
    const MKL_INT batch_size, const MKL_INT M, const MKL_INT N, const MKL_INT K, const double alpha,
    const double** A, const MKL_INT lda, const double** B, const MKL_INT ldb, const double beta,
    double** C, const MKL_INT ldc) {
  auto transa_cblas = to_cblas(trans_A);
  auto transb_cblas = to_cblas(trans_B);
  cblas_dgemm_batch(CblasColMajor, &transa_cblas, &transb_cblas, &M, &N, &K, &alpha,
                    A, &lda, B, &ldb, &beta, C, &ldc, 1, &batch_size);
}

void mkl_gemm_batched(
    const TransposeType trans_A, const TransposeType trans_B,
    const MKL_INT batch_size, const MKL_INT M, const MKL_INT N, const MKL_INT K, const c10::complex<float> alpha,
    const c10::complex<float>** A, const MKL_INT lda, const c10::complex<float>** B, const MKL_INT ldb,
    const c10::complex<float> beta, c10::complex<float>** C, const MKL_INT ldc) {
  auto transa_cblas = to_cblas(trans_A);
  auto transb_cblas = to_cblas(trans_B);
  cblas_cgemm_batch(CblasColMajor, &transa_cblas, &transb_cblas, &M, &N, &K,
                    reinterpret_cast<const void*>(&alpha),
                    reinterpret_cast<const void**>(A), &lda, reinterpret_cast<const void**>(B), &ldb,
                    reinterpret_cast<const void*>(&beta), reinterpret_cast<void**>(C), &ldc, 1, &batch_size);
}

void mkl_gemm_batched(
    const TransposeType trans_A, const TransposeType trans_B,
    const MKL_INT batch_size, const MKL_INT M, const MKL_INT N, const MKL_INT K, const c10::complex<double> alpha,
    const c10::complex<double>** A, const MKL_INT lda, const c10::complex<double>** B, const MKL_INT ldb,
    const c10::complex<double> beta, c10::complex<double>** C, const MKL_INT ldc) {
  auto transa_cblas = to_cblas(trans_A);
  auto transb_cblas = to_cblas(trans_B);
  cblas_zgemm_batch(CblasColMajor, &transa_cblas, &transb_cblas, &M, &N, &K,
                    reinterpret_cast<const void*>(&alpha),
                    reinterpret_cast<const void**>(A), &lda, reinterpret_cast<const void**>(B), &ldb,
                    reinterpret_cast<const void*>(&beta), reinterpret_cast<void**>(C), &ldc, 1, &batch_size);
}

void mkl_gemm_bf16bf16f32(
    TransposeType trans_A, TransposeType trans_B,
    MKL_INT M, MKL_INT N, MKL_INT K, const float alpha,
    const c10::BFloat16* A, MKL_INT lda, const c10::BFloat16* B, MKL_INT ldb,
    const float beta, float* C, MKL_INT ldc) {
#ifdef MKL_HAS_SBGEMM
  auto transa_cblas = to_cblas(trans_A);
  auto transb_cblas = to_cblas(trans_B);
  cblas_gemm_bf16bf16f32(CblasColMajor, transa_cblas, transb_cblas, M, N, K, alpha,
                         (const MKL_BF16*)A, lda, (const MKL_BF16*)B, ldb, beta, C, ldc);
#else
  TORCH_INTERNAL_ASSERT(false, "mkl_gemm_bf16bf16f32 requires mkl version > 2021.0");
#endif
}

void mkl_gemm_f16f16f32(
    TransposeType trans_A, TransposeType trans_B,
    int M, int N, int K, const float alpha,
    const c10::Half* A, int lda, const c10::Half* B, int ldb,
    const float beta, float* C, int ldc) {
#ifdef MKL_HAS_SHGEMM
  auto transa_cblas = to_cblas(trans_A);
  auto transb_cblas = to_cblas(trans_B);
  cblas_gemm_f16f16f32(CblasColMajor, transa_cblas, transb_cblas, M, N, K, alpha,
                         (const MKL_F16*)A, lda, (const MKL_F16*)B, ldb, beta, C, ldc);
#else
  TORCH_INTERNAL_ASSERT(false, "mkl_gemm_f16f16f32 requires mkl version >= 2024.0");
#endif
}

}} // namespace at::native

#endif
C10_DIAGNOSTIC_POP()
