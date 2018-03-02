#include "ATen/ATen.h"

#include <tuple>

namespace at {
namespace native{

std::tuple<Tensor, Tensor>
unique_cuda(const Tensor& self, const bool sorted, const bool return_inverse) {
  throw std::runtime_error(
      "unique is currently CPU-only, and lacks CUDA support. "
      "Pull requests welcome!");
  return std::make_tuple(self.type().tensor({0}), self.type().tensor({0}));
}

}  // namespace native
}  // namespace at
