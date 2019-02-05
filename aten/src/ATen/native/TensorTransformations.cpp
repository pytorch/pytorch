#include <ATen/native/TensorTransformations.h>
#include <ATen/WrapDimUtilsMulti.h>

#include <ATen/NativeFunctions.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <vector>

namespace at {
namespace native {

constexpr size_t dim_bitset_size = 64;

template <typename scalar_t>
void inline flip_cpu_kernel(
  const int64_t total_dims,
  const std::vector<int64_t>& stride_contiguous_v,
  const std::bitset<dim_bitset_size>& flip_dims_b,
  const Tensor& in_tensor,
  Tensor& out_tensor
){
  int64_t i;
  const int64_t numel = in_tensor.numel();
  const scalar_t* in_tensor_d = in_tensor.data<scalar_t>();
  scalar_t* out_tensor_d = out_tensor.data<scalar_t>();
  auto sizes_v = in_tensor.sizes().vec();
  auto strides_v = in_tensor.strides().vec();

  #pragma omp parallel for private(i) if (numel > 1000)
  for (i = 0; i < numel; i++) {
    int64_t cur_indices = i;
    int64_t rem = 0;
    int64_t dst_offset = 0;

    for (int64_t d = 0; d < total_dims; d++) {
      int64_t temp = cur_indices;
      cur_indices = cur_indices / stride_contiguous_v[d];
      rem = temp - cur_indices * stride_contiguous_v[d];
      dst_offset += flip_dims_b[d] ? (sizes_v[d] - 1 - cur_indices) * strides_v[d] : cur_indices * strides_v[d];
      cur_indices = rem;
    }
    out_tensor_d[i] = in_tensor_d[dst_offset];
  }
}

Tensor flip_cpu(const Tensor& self, IntArrayRef dims) {
  auto in_tensor = self;
  const int64_t total_dims = in_tensor.dim();
  auto flip_dims_b = at::dim_list_to_bitset(dims, total_dims);
  Tensor out_tensor = at::empty_like(in_tensor);

  // create contiguous strides for input tensor
  auto stride_contiguous_v = std::vector<int64_t>(total_dims);
  for (int64_t i = total_dims - 1; i >= 0; i--) {
    if (i == total_dims - 1) {
      stride_contiguous_v[i] = 1;
    } else {
      stride_contiguous_v[i] = std::max<int64_t>(in_tensor.size(i + 1), 1) * stride_contiguous_v[i + 1];
    }
  }

  AT_DISPATCH_ALL_TYPES(in_tensor.type(), "flip_cpu", [&] {
    flip_cpu_kernel<scalar_t>(
      total_dims,
      stride_contiguous_v,
      flip_dims_b,
      in_tensor,
      out_tensor
    );
  });

  return out_tensor;
}

Tensor roll_cpu(const Tensor& self, IntArrayRef shifts, IntArrayRef dims) {
  if (dims.size() != 1 || shifts.size() != 1) {
    return roll_common(self, shifts, dims);
  }
  // avoid a div zero error below.
  if (self.numel() == 0) {
    return self.clone();
  }
  int64_t dim = dims[0];
  int64_t size = self.size(dim);
  int64_t start = (size - shifts[0]) % size;
  // Behavior of % is different in C++ vs Python for negative numbers. This
  // corrects the difference.
  if (start < 0) {
    start = start + size;
  }
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

Tensor rot90(const Tensor& self, int64_t k, IntArrayRef dims) {
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
