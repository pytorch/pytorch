#include "TensorTransformations.h"

#include "ATen/NativeFunctions.h"

namespace at {
namespace native {

Tensor flip_cpu(const Tensor& self, IntList dims) {
  int64_t total_dims = self.dim(), flip_dims_size = dims.size();
  check_errors(total_dims, flip_dims_size, dims);

  auto indices = std::vector<at::Tensor>(flip_dims_size);
  for (int64_t i = 0; i < flip_dims_size; i++) {
    indices[i] = at::arange(self.type().toScalarType(at::ScalarType::Long), self.size(i) - 1, -1, -1);
  }
  // creates a meshgrid
  for (int64_t i = 0; i < flip_dims_size; i++) {
    auto temp = std::vector<int64_t>(flip_dims_size, 1);
    temp[i] = indices[i].size(0);
    indices[i] = indices[i].view(IntList(temp));
  }
  return self.index(TensorList(indices));
}

}} // namespace at::native
