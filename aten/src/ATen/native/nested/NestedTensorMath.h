#pragma once

#include <c10/macros/Macros.h>

#include <vector>

namespace at {
namespace native {
struct NestedTensorImpl;

// TODO: cache this and only do it once per NestedTensor
int64_t get_consistent_last_dim_of_nested_tensor(const NestedTensorImpl& nt);

TORCH_API std::vector<int64_t> NestedTensor_get_max_size(const NestedTensorImpl& nt);

} // namespace native
} // namespace at
