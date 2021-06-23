#include <ATen/Operators.h>

namespace at { namespace _ops {

Tensor & requires_grad_(Tensor & self, bool requires_grad) {
  self.requires_grad_(requires_grad);
  return self;
}

${definitions}

}} // namespace at::_ops
