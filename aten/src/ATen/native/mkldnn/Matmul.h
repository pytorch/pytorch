#pragma once

#include <ATen/ATen.h>
#include <ATen/Config.h>

namespace at { namespace native {

// result = beta * result + alpha * gemm(mat1, mat2)
// need mat, mat2 to be 2-D or 3-D Tensors
TORCH_API void mkldnn_matmul(
        const Tensor &mat1,
        const Tensor &mat2,
        const Tensor &result,
        float beta=1,
        float alpha=1);

}}
