#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

namespace at { namespace native {

static const double SELU_ALPHA = 1.6732632423543772848170429916717;
static const double SELU_SCALE = 1.0507009873554804934193349852946;

Tensor selu(const Tensor & self) {
  return at::elu(self, SELU_ALPHA, SELU_SCALE);
}

Tensor & selu_(Tensor & self) {
  // TODO: at::elu_ should return `Tensor &`
  at::elu_(self, SELU_ALPHA, SELU_SCALE);
  return self;
}

}}  // namespace at::native
