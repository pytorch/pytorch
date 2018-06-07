#include "TensorTransformations.h"

#include "ATen/NativeFunctions.h"

namespace at {
namespace native {

Tensor flip_cpu(const Tensor& self, IntList dims) {
  const int64_t total_dims = self.dim(), flip_dims_size = dims.size();
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

  /*TODO:
  Translate this:
  ----------------
  multi_indices = multi_meshgrid(*indices)
      final_indices = [slice(i) for i in tensor.shape]
      for i, dim in enumerate(dims):
          final_indices[dim] = multi_indices[i]
      return tensor[final_indices]
  ---------------
  Permute results:
  ---------------
  result = b.permute(0, 2, 1)
  --------------
  */

  // auto dims_v = std::vector<int64_t>(dims);
  // auto permute_order = std::vector<int64_t>(dims);
  // for (int64_t i = 0; i < total_dims; i++) {
  //   if (std::find(dims_v.begin(), dims_v.end(), i) == dims_v.end()) {
  //     permute_order.emplace_back(i);
  //   }
  // }
  //
  // for (int64_t i = 0; i < total_dims; i++) {
  //   printf("i=%ld, permute=%ld\n", i, permute_order[i]);
  // }
  //
  // auto out_tensor = self.index(TensorList(indices));
  // return out_tensor.permute(IntList(permute_order));

  auto out_tensor = self.index(TensorList(indices));
  return out_tensor;
}

}} // namespace at::native
