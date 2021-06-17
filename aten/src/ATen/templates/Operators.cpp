#include <ATen/Operators.h>
#include <ATen/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at { namespace _ops {

Tensor & requires_grad_(Tensor & self, bool requires_grad) {
  self.requires_grad_(requires_grad);
  return self;
}

${definitions}

}} // namespace at::_ops
