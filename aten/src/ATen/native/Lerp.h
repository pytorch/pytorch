#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

using lerp_fn_scalar = void (*)(
    at::Tensor& ret,
    const at::Tensor& self,
    const at::Tensor& end,
    const Scalar& weight);

using lerp_fn_tensor = void (*)(
    at::Tensor& ret,
    const at::Tensor& self,
    const at::Tensor& end,
    const at::Tensor& weights);

DECLARE_DISPATCH(lerp_fn_scalar, lerp_kernel_scalar_weight);
DECLARE_DISPATCH(lerp_fn_tensor, lerp_kernel_tensor_weight);

} // namespace native
} // namespace at
