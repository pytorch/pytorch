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

bool use_mkldnn_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result_opt);

// Try running mkldnn optimized gemm, or returns false if naive gemm would be faster
template <typename scalar_t>
bool mkldnn_gemm(
    TransposeType transa,
    TransposeType transb,
    int64_t m,
    int64_t n,
    int64_t k,
    float alpha,
    const scalar_t* a,
    int64_t lda,
    const scalar_t* b,
    int64_t ldb,
    float beta,
    scalar_t* c,
    int64_t ldc);
}

}
