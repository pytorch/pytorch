#include "ATen/ATen.h"

#include <tuple>
A
namespace at {
namespace native{

std::tuple<Tensor, Tensor>
unique_cuda(const Tensor& self, const bool sorted, const bool return_inverse) {
  throw std::runtime_error(
      "unique is currently CPU-only, and lacks CUDA support. "
      "Pull requests welcome!");
}

}  // namespace native
}  // namespace at
