#pragma once

#include <c10/macros/Macros.h>
#include <ATen/NestedTensorImpl.h>

#include <vector>

namespace at {
namespace native {
struct NestedTensorImpl;

// TODO: cache this and only do it once per NestedTensor
int64_t get_consistent_last_dim_of_nested_tensor(const NestedTensorImpl& nt);

inline std::vector<int64_t> NestedTensor_get_max_size_from_size_tensor(const Tensor& sizes) {
  if (sizes.dim() == 0) {
    return {};
  }
  const auto sizes_ptr = sizes.data_ptr<int64_t>();
  const auto sizes_size_0 = sizes.sizes()[0];
  const auto sizes_size_1 = sizes.sizes()[1];
  TORCH_INTERNAL_ASSERT(sizes_size_1 > 0);
  std::vector<int64_t> results(sizes_size_1, 0);
  for (const auto ii : c10::irange(sizes_size_0)) {
    for (const auto jj : c10::irange(sizes_size_1)) {
      auto val = sizes_ptr[ii * sizes_size_1 + jj];
      if (results[jj] < val) {
        results[jj] = val;
      }
    }
  }
  return results;
}

inline std::vector<int64_t> NestedTensor_get_max_size(const NestedTensorImpl& nt) {
  const auto& sizes = nt.get_nested_size_tensor();
  return NestedTensor_get_max_size_from_size_tensor(sizes);
}

Tensor NestedTensor_to_padded_tensor_generic(const Tensor& t, double padding);

} // namespace native
} // namespace at
