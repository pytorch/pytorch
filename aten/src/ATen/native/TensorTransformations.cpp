#include "ATen/native/TensorTransformations.h"

#include <ATen/NativeFunctions.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <vector>

namespace at {
namespace native {

Tensor flip_cpu(const Tensor& self, IntList dims) {
  auto in_tensor = self;
  const int64_t total_dims = self.dim();
  const int64_t numel = self.numel();
  auto strides = self.strides();
  auto sizes = self.sizes();
  const int64_t flip_dims_size = dims.size();
  auto flip_dims_v = dims.vec();

  flip_check_errors(total_dims, flip_dims_size, dims);
  wrap_all_dims(flip_dims_v, total_dims);
  // std::sort(flip_dims_v.begin(), flip_dims_v.end());
  // auto final_indices = std::vector<at::Tensor>(total_dims);
  //
  // auto indices = std::vector<at::Tensor>(flip_dims_size);
  // for (int64_t i = 0; i < flip_dims_size; i++) {
  //   indices[i] = at::arange(self.size(flip_dims_v[i]) - 1, -1, -1, self.type().toScalarType(at::kLong));
  //   // creates a meshgrid
  //   auto temp = std::vector<int64_t>(flip_dims_size, 1);
  //   temp[i] = indices[i].size(0);
  //   indices[i] = indices[i].view(IntList(temp));
  //   final_indices[flip_dims_v[i]] = indices[i];
  // }
  //
  // // check if distance between two flip dims >= 2, where permute of output tensor is needed,
  // // because the advanced indexing puts all non-consecutive indices in the beginning of the tensor
  // bool to_permute = false;
  // int64_t first = flip_dims_v[0], second = flip_dims_v[0];
  // for (int64_t i = 1; i < flip_dims_size; i++) {
  //   second = flip_dims_v[i];
  //   if (second - first >= 2) {
  //     to_permute = true;
  //     break;
  //   }
  //   first = second;
  // }
  //
  // if (to_permute) {
  //   // permute output tensor
  //   auto permute_order = std::vector<int64_t>(flip_dims_v);
  //   for (int64_t i = 0; i < total_dims; i++) {
  //     if (std::find(flip_dims_v.begin(), flip_dims_v.end(), i) == flip_dims_v.end()) {
  //       permute_order.emplace_back(i);
  //     }
  //   }
  //   auto out_tensor = self.index(TensorList(final_indices));
  //   return out_tensor.permute(IntList(permute_order));
  // }
  //
  // auto out_tensor = self.index(TensorList(final_indices));

  Tensor out_tensor = at::empty_like(in_tensor);
  Tensor stride_contiguous = at::zeros({total_dims}, kLong);
  int64_t* stride_contiguous_d = stride_contiguous.data<int64_t>();
  for (int64_t i = total_dims - 1; i >= 0; i--) {
    if (i == total_dims - 1) {
      stride_contiguous_d[i] = 1;
    } else {
      stride_contiguous_d[i] = std::max<int64_t>(sizes[i+1], 1) * stride_contiguous_d[i + 1];
    }
  }

  AT_DISPATCH_ALL_TYPES(in_tensor.type(), "flip_cpu", [&] {

    auto out_tensor_d = out_tensor.data<scalar_t>();
    auto in_tensor_d = in_tensor.data<scalar_t>();

    for (int64_t i = 0; i < numel; i++) {
      int64_t cur_indices = i;
      int64_t rem = 0;
      int64_t dst_offset = 0;

      for (int64_t d = 0; d < total_dims; d++) {
        int64_t temp = cur_indices;
        cur_indices = cur_indices / stride_contiguous_d[d];
        rem = temp - cur_indices * stride_contiguous_d[d];
        // flip the indices if it is in flip_dims
        for (auto fd : flip_dims_v) {
          if (d == fd) cur_indices = sizes[d] - 1 - cur_indices;
        }
        dst_offset += cur_indices * strides[d];
        cur_indices = rem;
      }
      out_tensor_d[i] = in_tensor_d[dst_offset];
    }
  });

  return out_tensor;
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
