#include <ATen/ATen.h>

namespace at {
namespace native {

Tensor _noop_unary(const Tensor& self) {
  return self;
}

Tensor _noop_unary_manual(const Tensor& self) {
  return self;
}

Tensor _noop_binary(const Tensor& self, const Tensor& other) {
  return self;
}

Tensor _noop_binary_manual(const Tensor& self, const Tensor& other) {
  return self;
}

}  // namespace native
}  // namespace at
