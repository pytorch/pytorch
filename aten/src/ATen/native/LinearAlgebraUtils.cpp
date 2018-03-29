#include "ATen/native/LinearAlgebraUtils.h"

namespace at { namespace native {

Tensor cloneBatchedColumnMajor(const Tensor& src) {
  auto second_to_last_dim = src.dim() - 2;
  auto last_dim = src.dim() - 1;

  auto desired_strides = defaultStrides(src.sizes());
  desired_strides[last_dim] = src.size(second_to_last_dim);
  desired_strides[second_to_last_dim] = 1;

  // Fast path for if the input already has correct strides
  if (std::equal(desired_strides.begin(), desired_strides.end(), src.strides().begin())) {
    auto result = src.transpose(-2, -1).clone();
    result.transpose_(-2, -1);
    return result;
  }

  // Slow path: copy with desired strides
  auto result = src.type().tensor(src.sizes(), desired_strides);
  result.copy_(src);
  return result;
}

}}  // namespace at::native
