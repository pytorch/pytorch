#pragma once
#include <ATen/native/TransposeType.h>
#include <c10/util/complex.h>
#include <c10/core/ScalarType.h>

namespace at {
namespace native {

void mkl_gemm_batched(
    TransposeType trans_A, TransposeType trans_B,
    int batch_size, int M, int N, int K, float alpha,
    const float** A, int lda, const float** B, int ldb, float beta,
    float** C, int ldc);

void mkl_gemm_batched(
    TransposeType trans_A, TransposeType trans_B,
    int batch_size, int M, int N, int K, double alpha,
    const double** A, int lda, const double** B, int ldb, double beta,
    double** C, int ldc);

void mkl_gemm_batched(
    TransposeType trans_A, TransposeType trans_B,
    int batch_size, int M, int N, int K, c10::complex<float> alpha,
    const c10::complex<float>** A, int lda, const c10::complex<float>** B, int ldb,
    c10::complex<float> beta, c10::complex<float>** C, int ldc);

void mkl_gemm_batched(
    TransposeType trans_A, TransposeType trans_B,
    int batch_size, int M, int N, int K, c10::complex<double> alpha,
    const c10::complex<double>** A, int lda, const c10::complex<double>** B, int ldb,
    c10::complex<double> beta, c10::complex<double>** C, int ldc);

void mkl_gemm_bf16bf16f32(
    TransposeType trans_A, TransposeType trans_B,
    int M, int N, int K, const float alpha,
    const c10::BFloat16* A, int lda, const c10::BFloat16* B, int ldb,
    const float beta, float* C, int ldc);

}}  // namespace at::native
