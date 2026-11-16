#pragma once
#include <ATen/Config.h>
#include <ATen/native/TransposeType.h>
#include <c10/util/complex.h>
#include <c10/core/ScalarType.h>

#if !AT_MKL_ENABLED()
#define MKL_INT int
#else
#include <mkl.h>
#endif

namespace at::native {

void mkl_gemm_batched(
    TransposeType trans_A, TransposeType trans_B,
    MKL_INT batch_size, MKL_INT M, MKL_INT N, MKL_INT K, float alpha,
    const float** A, MKL_INT lda, const float** B, MKL_INT ldb, float beta,
    float** C, MKL_INT ldc);

void mkl_gemm_batched(
    TransposeType trans_A, TransposeType trans_B,
    MKL_INT batch_size, MKL_INT M, MKL_INT N, MKL_INT K, double alpha,
    const double** A, MKL_INT lda, const double** B, MKL_INT ldb, double beta,
    double** C, MKL_INT ldc);

void mkl_gemm_batched(
    TransposeType trans_A, TransposeType trans_B,
    MKL_INT batch_size, MKL_INT M, MKL_INT N, MKL_INT K, c10::complex<float> alpha,
    const c10::complex<float>** A, MKL_INT lda, const c10::complex<float>** B, MKL_INT ldb,
    c10::complex<float> beta, c10::complex<float>** C, MKL_INT ldc);

void mkl_gemm_batched(
    TransposeType trans_A, TransposeType trans_B,
    MKL_INT batch_size, MKL_INT M, MKL_INT N, MKL_INT K, c10::complex<double> alpha,
    const c10::complex<double>** A, MKL_INT lda, const c10::complex<double>** B, MKL_INT ldb,
    c10::complex<double> beta, c10::complex<double>** C, MKL_INT ldc);

void mkl_gemm_bf16bf16f32(
    TransposeType trans_A, TransposeType trans_B,
    MKL_INT M, MKL_INT N, MKL_INT K, const float alpha,
    const c10::BFloat16* A, MKL_INT lda, const c10::BFloat16* B, MKL_INT ldb,
    const float beta, float* C, MKL_INT ldc);

void mkl_gemm_f16f16f32(
    TransposeType trans_A, TransposeType trans_B,
    int M, int N, int K, const float alpha,
    const c10::Half* A, int lda, const c10::Half* B, int ldb,
    const float beta, float* C, int ldc);
}  // namespace at::native
