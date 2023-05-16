#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/native/LinearAlgebraUtils.h>  // For TransposeType

namespace at { namespace native {

// result = beta * result + alpha * gemm(mat1, mat2)
TORCH_API void mkldnn_matmul(
        const Tensor &mat1,
        const Tensor &mat2,
        const Tensor &result,
        float beta=1,
        float alpha=1);

bool use_mkldnn_bf16_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result_opt);

bool use_mkldnn_fp16_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result_opt);

// Try running mkldnn optimized gemm, or returns false if naive gemm would be faster
bool mkldnn_bf16_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const c10::BFloat16 *a, int64_t lda,
    const c10::BFloat16 *b, int64_t ldb,
    float beta,
    c10::BFloat16 *c, int64_t ldc);

bool use_mkldnn_lower_precision_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result);

}

}
