#pragma once

#include <ATen/ATen.h>
#include <ATen/Config.h>

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

}

}
