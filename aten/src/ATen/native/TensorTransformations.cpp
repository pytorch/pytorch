#include "ATen/native/TensorTransformations.h"

#include <ATen/NativeFunctions.h>
#include <ATen/Error.h>

#include <algorithm>
#include <vector>

namespace at {
namespace native {

Tensor flip_cpu(const Tensor& self, IntList dims) {
  const int64_t total_dims = self.dim(), flip_dims_size = dims.size();
  check_errors(total_dims, flip_dims_size, dims);

  auto flip_dims_v = std::vector<int64_t>(dims);
  std::sort(flip_dims_v.begin(), flip_dims_v.end());
  auto final_indices = std::vector<at::Tensor>(total_dims);

  auto indices = std::vector<at::Tensor>(flip_dims_size);
  for (int64_t i = 0; i < flip_dims_size; i++) {
    indices[i] = at::arange(self.size(flip_dims_v[i]) - 1, -1, -1, self.type().toScalarType(at::kLong));
    // creates a meshgrid
    auto temp = std::vector<int64_t>(flip_dims_size, 1);
    temp[i] = indices[i].size(0);
    indices[i] = indices[i].view(IntList(temp));
    final_indices[flip_dims_v[i]] = indices[i];
  }

  // check if distance between two flip dims >= 2, where permute of output tensor is needed,
  // because the advanced indexing puts all non-consecutive indices in the beginning of the tensor
  bool to_permute = false;
  int64_t first = flip_dims_v[0], second = flip_dims_v[0];
  for (int64_t i = 1; i < flip_dims_size; i++) {
    second = flip_dims_v[i];
    if (second - first >= 2) {
      to_permute = true;
      break;
    }
    first = second;
  }

  if (to_permute) {
    // permute output tensor
    auto permute_order = std::vector<int64_t>(flip_dims_v);
    for (int64_t i = 0; i < total_dims; i++) {
      if (std::find(flip_dims_v.begin(), flip_dims_v.end(), i) == flip_dims_v.end()) {
        permute_order.emplace_back(i);
      }
    }
    auto out_tensor = self.index(TensorList(final_indices));
    return out_tensor.permute(IntList(permute_order));
  }

  auto out_tensor = self.index(TensorList(final_indices));
  return out_tensor;
}

}} // namespace at::native
