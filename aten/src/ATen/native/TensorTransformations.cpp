#include <ATen/native/TensorTransformations.h>

#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <vector>

namespace at {
namespace native {

Tensor flip(const Tensor& self, IntArrayRef dims) {
  auto in_tensor = self;
  const int64_t total_dims = in_tensor.dim();
  auto flip_dims_b = at::dim_list_to_bitset(dims, total_dims);
  Tensor out_tensor = at::empty_like(in_tensor, MemoryFormat::Contiguous);
  int n = 0;
  auto strides = out_tensor.strides().vec();
  for(int64_t i = 0; i < total_dims; i++) {
    if(flip_dims_b[i] && out_tensor.size(i) > 1) {
      n++;
      strides[i] = 0;
    }
  }

  // Nothing to do, we return fast
  if (n == 0) {
    out_tensor.copy_(self);
    return out_tensor;
  }

  //create dummy output with 0 strides at flipped dimension, to prevent tensorIterator from coalescing flipped dims
  auto restrided_out = out_tensor.as_strided(out_tensor.sizes(), strides);
  TensorIteratorConfig config;
  config.set_check_mem_overlap(false)
        .check_all_same_dtype(false)
        .declare_static_dtype_and_device(self.scalar_type(), self.device())
        .add_output(out_tensor)
        .add_input(self)
        .add_input(restrided_out);
  auto iter = config.build();
  iter.flip_strides(0, 2); //flip strides of the real output using dummy output as model (dimension should be flipped where stride is 0)

  flip_stub(iter.device_type(), iter);

  return out_tensor;
}

Tensor roll_cpu(const Tensor& self, IntArrayRef shifts, IntArrayRef dims) {
  if (dims.size() != 1 || shifts.size() != 1) {
    return roll_common(self, shifts, dims);
  }
  // avoid a div zero error below.
  if (self.numel() == 0) {
    return self.clone(at::MemoryFormat::Preserve);
  }
  int64_t dim = dims[0];
  int64_t size = self.size(dim);
  int64_t start = (size - shifts[0]) % size;
  // Behavior of % is different in C++ vs Python for negative numbers. This
  // corrects the difference.
  if (start < 0) {
    start = start + size;
  }
  auto t0 = self.narrow(dim, start, size-start);
  auto t1 = self.narrow(dim, 0, start);
  return at::cat({t0, t1}, dim);
}

Tensor rot90(const Tensor& self, int64_t k, IntArrayRef dims) {
  const int64_t total_dims = self.dim(), total_rot_dims = dims.size();

  TORCH_CHECK(total_rot_dims == 2,
    "expected total rotation dims == 2, but got dims = ", total_rot_dims);

  TORCH_CHECK(total_dims >= 2,
    "expected total dims >= 2, but got total dims = ", total_dims);

  TORCH_CHECK(dims[0] != dims[1] && std::abs(dims[0] - dims[1]) != total_dims,
    "expected rotation dims to be different, but got dim0 = ", dims[0],
    " and dim1 = ", dims[1]);

  // check range of dims
  TORCH_CHECK(dims[0] < total_dims && dims[0] >= -total_dims,
    "Rotation dim0 out of range, dim0 = ", dims[0]);

  TORCH_CHECK(dims[1] < total_dims && dims[1] >= -total_dims,
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
      return self.clone(at::MemoryFormat::Contiguous);
  }
}

Tensor fliplr(const Tensor& self) {
  TORCH_CHECK(self.dim() >= 2, "Input must be >= 2-d.");

  return self.flip({1});
}

Tensor flipud(const Tensor& self) {
  TORCH_CHECK(self.dim() >= 1, "Input must be >= 1-d.");

  return self.flip({0});
}

Tensor atleast_1d(const Tensor& self) {
  switch (self.dim()) {
    case 0:
      return self.reshape({1});
    default:
      return self;
  }
}

std::vector<Tensor> atleast_1d(TensorList tensors) {
  std::vector<Tensor> result(tensors.size());
  auto transform_lambda = [](const Tensor& input) -> Tensor {
    return at::native::atleast_1d(input);
  };
  std::transform(tensors.cbegin(), tensors.cend(), result.begin(), transform_lambda);
  return result;
}

Tensor atleast_2d(const Tensor& self) {
  switch (self.dim()) {
    case 0:
      return self.reshape({1, 1});
    case 1: {
      return self.unsqueeze(0);
    }
    default:
      return self;
  }
}

std::vector<Tensor> atleast_2d(TensorList tensors) {
  std::vector<Tensor> result(tensors.size());
  auto transform_lambda = [](const Tensor& input) -> Tensor {
    return at::native::atleast_2d(input);
  };
  std::transform(tensors.cbegin(), tensors.cend(), result.begin(), transform_lambda);
  return result;
}

Tensor atleast_3d(const Tensor& self) {
  switch (self.dim()) {
    case 0:
      return self.reshape({1, 1, 1});
    case 1: {
      return self.unsqueeze(0).unsqueeze(-1);
    }
    case 2: {
      return self.unsqueeze(-1);
    }
    default:
      return self;
  }
}

std::vector<Tensor> atleast_3d(TensorList tensors) {
  std::vector<Tensor> result(tensors.size());
  auto transform_lambda = [](const Tensor& input) -> Tensor {
    return at::native::atleast_3d(input);
  };
  std::transform(tensors.cbegin(), tensors.cend(), result.begin(), transform_lambda);
  return result;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(flip_stub);

}} // namespace at::native
