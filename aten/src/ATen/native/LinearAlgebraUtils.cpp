#include "ATen/native/LinearAlgebraUtils.h"

namespace at { namespace native {

Tensor cloneBatchedColumnMajor(const Tensor& src) {
  auto second_to_last_dim = src.ndimension() - 2;
  auto last_dim = src.ndimension() - 1;

  auto desired_strides = defaultStrides(src.sizes());
  desired_strides[last_dim] = src.size(second_to_last_dim);
  desired_strides[second_to_last_dim] = 1;

  auto result = src.type().tensor(src.sizes(), desired_strides);
  result.copy_(src);
  return result;
}

}}  // namespace at::native
