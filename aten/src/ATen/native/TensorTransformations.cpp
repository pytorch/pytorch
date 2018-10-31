#include "ATen/native/TensorTransformations.h"

#include <ATen/NativeFunctions.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <vector>

namespace at {
namespace native {

template <typename scalar_t>
void inline flip_cpu_kernel(
  const int64_t numel,
  const int64_t total_dims,
  const int64_t flip_dims_size,
  const int64_t* stride_contiguous_d,
  const int64_t* flip_dims_d,
  const int64_t* strides_d,
  const int64_t* sizes_d,
  const scalar_t* in_tensor_d,
  scalar_t* out_tensor_d
){
  int64_t i;

  #pragma omp parallel for private(i) if (numel > 1000)
  for (i = 0; i < numel; i++) {
    int64_t cur_indices = i;
    int64_t rem = 0;
    int64_t dst_offset = 0;

    for (int64_t d = 0; d < total_dims; d++) {
      int64_t temp = cur_indices;
      cur_indices = cur_indices / stride_contiguous_d[d];
      rem = temp - cur_indices * stride_contiguous_d[d];
      // flip the indices if it is in flip_dims
      for (int64_t j = 0; j < flip_dims_size; j++) {
        if (d == flip_dims_d[j]) cur_indices = sizes_d[d] - 1 - cur_indices;
      }
      dst_offset += cur_indices * strides_d[d];
      cur_indices = rem;
    }
    out_tensor_d[i] = in_tensor_d[dst_offset];
  }
}

Tensor flip_cpu(const Tensor& self, IntList dims) {
  auto in_tensor = self;
  const int64_t total_dims = self.dim();
  const int64_t numel = self.numel();

  auto strides = self.strides();
  auto strides_v = strides.vec();
  auto strides_t = at::CPU(kLong).tensorFromBlob(strides_v.data(), {static_cast<int64_t>(strides_v.size())});

  auto sizes = self.sizes();
  auto sizes_v = sizes.vec();
  auto sizes_t = at::CPU(kLong).tensorFromBlob(sizes_v.data(), {static_cast<int64_t>(sizes_v.size())});

  const int64_t flip_dims_size = dims.size();
  auto flip_dims_v = dims.vec();
  auto flip_dims_t = at::CPU(kLong).tensorFromBlob(flip_dims_v.data(), {static_cast<int64_t>(flip_dims_v.size())});

  flip_check_errors(total_dims, flip_dims_size, dims);
  wrap_all_dims(flip_dims_v, total_dims);

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

    flip_cpu_kernel<scalar_t>(
      numel,
      total_dims,
      flip_dims_size,
      stride_contiguous_d,
      flip_dims_t.data<int64_t>(),
      strides_t.data<int64_t>(),
      sizes_t.data<int64_t>(),
      in_tensor_d,
      out_tensor_d
    );
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
