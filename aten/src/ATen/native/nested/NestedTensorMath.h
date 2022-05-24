#pragma once

#include <c10/macros/Macros.h>
#include <ATen/NestedTensorImpl.h>

#include <vector>

namespace at {
namespace native {
struct NestedTensorImpl;

// TODO: cache this and only do it once per NestedTensor
int64_t get_consistent_last_dim_of_nested_tensor(const NestedTensorImpl& nt);

TORCH_API std::vector<int64_t> NestedTensor_get_max_size(const NestedTensorImpl& nt);

TORCH_API Tensor NestedTensor_to_padded_tensor_generic(const Tensor& t, double padding, OptionalIntArrayRef output_size);

int64_t NestedTensor_size_int(const Tensor& self, int64_t d);
std::vector<at::Tensor> nested_size(const Tensor& self);
at::Tensor NestedTensor_nested_size_tensor(const Tensor& self);

} // namespace native
} // namespace at
