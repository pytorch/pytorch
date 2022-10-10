#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ATen_fwd.h>
#include <c10/macros/Macros.h>
namespace at {
namespace native {

TORCH_API Tensor NestedTensor_to_padded_tensor_generic(
    const Tensor& t,
    double padding,
    OptionalIntArrayRef output_size);

} // namespace native
} // namespace at
