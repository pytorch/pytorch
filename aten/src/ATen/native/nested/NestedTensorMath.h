#pragma once

#include <ATen/NestedTensorImpl.h>
#include <c10/macros/Macros.h>

#include <vector>

namespace at {
namespace native {
struct NestedTensorImpl;

// TODO: cache this and only do it once per NestedTensor
int64_t get_consistent_last_dim_of_nested_tensor(const NestedTensorImpl& nt);

TORCH_API std::vector<int64_t> NestedTensor_get_max_size(
    const NestedTensorImpl& nt);

TORCH_API Tensor NestedTensor_to_padded_tensor_generic(
    const Tensor& t,
    double padding,
    OptionalIntArrayRef output_size);

} // namespace native
} // namespace at
