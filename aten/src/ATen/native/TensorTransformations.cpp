#include "ATen/native/TensorTransformations.h"

#include <ATen/NativeFunctions.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <vector>

namespace at {
namespace native {

Tensor flip_cpu(const Tensor& self, IntList dims) {
  const int64_t total_dims = self.dim(), flip_dims_size = dims.size();
  flip_check_errors(total_dims, flip_dims_size, dims);

  auto flip_dims_v = dims.vec();
  wrap_all_dims(flip_dims_v, total_dims);
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

Tensor roll_cpu(const Tensor& self, IntList shifts, IntList dims) {
  // todo: support rolling along no or multiple dimensions as in numpy.roll.
  AT_CHECK(dims.size() == 1, "only single dimension roll currently supported");
  AT_CHECK(shifts.size() == dims.size(), "shifts and dimensions must align");
  // avoid a div zero error below.
  if( self.numel() == 0 ) {
    return self.clone();
  }
  int64_t dim = dims[0];
  int64_t size = self.size(dim);
  int64_t start = (size - shifts[0]) % size;
  // Behavior of % is different in C++ vs Python for negative numbers. This
  // corrects the difference.
  if( start < 0 ) start = start + size;

  auto tensors = self.unbind(dim);
  std::vector<Tensor> vec = std::vector<Tensor>(size);
  int64_t index = 0;
  for (int64_t i = start; i < size; i++) {
    vec[index++] = tensors[i];
  }

  for (int64_t i = 0; i < start; i++) {
    vec[index++] = tensors[i];
  }

  return at::stack(vec, dim);
}

Tensor rot90(const Tensor& self, int64_t k, IntList dims) {
  const int64_t total_dims = self.dim(), total_rot_dims = dims.size();

  AT_CHECK(total_rot_dims == 2,
    "expected total rotation dims == 2, but got dims = ", total_rot_dims);

  AT_CHECK(total_dims >= 2,
    "expected total dims >= 2, but got total dims = ", total_dims);

  AT_CHECK(dims[0] != dims[1] && std::abs(dims[0] - dims[1]) != total_dims,
    "expected rotation dims to be different, but got dim0 = ", dims[0],
    " and dim1 = ", dims[1]);

  // check range of dims
  AT_CHECK(dims[0] < total_dims && dims[0] >= -total_dims,
    "Rotation dim0 out of range, dim0 = ", dims[0]);

  AT_CHECK(dims[1] < total_dims && dims[1] >= -total_dims,
    "Rotation dim1 out of range, dim1 = ", dims[1]);

  // handle modulo with negative k
  k = (4 + (k % 4)) % 4;

  switch(k) {
    case 1:
      return self.flip({dims[1]}).transpose_(dims[0], dims[1]);
    case 2:
      return self.flip(dims);
    case 3:
      return self.flip({dims[0]}).transpose_(dims[0], dims[1]);
    default:
      return self.clone();
  }
}

}} // namespace at::native
