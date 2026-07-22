#pragma once

#include <c10/macros/Export.h>

#include <ATen/Tensor.h>
#include <ATen/core/Scalar.h>

namespace at::native::sparse {

TORCH_API void sparse_sampled_addmm_check_inputs(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result);

} // namespace at::native::sparse
