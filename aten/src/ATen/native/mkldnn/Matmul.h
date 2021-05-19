#pragma once

#include <ATen/ATen.h>
#include <ATen/Config.h>

namespace at { namespace native {

// result = beta * result + alpha * gemm(mat, mat2)
TORCH_API void mkldnn_matmul(
        const Tensor &mat1,
        const Tensor &mat2,
        Tensor &result,
        float beta=1,
        float alpha=1);

}}

