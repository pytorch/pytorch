#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/native/LinearAlgebraUtils.h>  // For TransposeType

namespace at::native {

// result = beta * result + alpha * gemm(mat1, mat2)
TORCH_API void onednn_matmul(
        const Tensor &mat1,
        const Tensor &mat2,
        const Tensor &result,
        float beta=1,
        float alpha=1);

bool use_onednn_bf16_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result_opt);

bool use_onednn_fp16_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result_opt);

bool use_onednn_bf32_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result_opt);

// Try running mkldnn optimized gemm, or returns false if naive gemm would be faster
bool onednn_bf16_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const c10::BFloat16 *a, int64_t lda,
    const c10::BFloat16 *b, int64_t ldb,
    float beta,
    c10::BFloat16 *c, int64_t ldc);

bool onednn_fp16_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const c10::Half *a, int64_t lda,
    const c10::Half *b, int64_t ldb,
    float beta,
    c10::Half *c, int64_t ldc);

/*
oneDNN implicit reduced precision arithmetic feature
https://github.com/mgouicem/oneDNN/tree/mgouicem/rfcs/implicit_downconvert/rfcs/20210301-computation-datatype
to allow implicitly cast data type from FP32 to BF16 in onednn compute primitives
*/
bool onednn_bf32_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const float *a, int64_t lda,
    const float *b, int64_t ldb,
    float beta,
    float *c, int64_t ldc);

bool use_onednn_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result);

// x:s8 * w:s8 -> y:s32
TORCH_API void onednn_matmul_i8i8i32(
    const Tensor &mat1,
    const Tensor &mat2,
    const Tensor &result);

}
