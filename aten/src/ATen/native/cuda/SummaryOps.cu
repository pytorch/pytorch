#include "ATen/ATen.h"

namespace at { namespace native {

Tensor _bincount_cuda(
    const Tensor& self,
    const Tensor& weights,
    int64_t minlength) {
  throw std::runtime_error(
      "bincount is currently CPU-only, and lacks CUDA support.");
}

}} // namespace at::native
