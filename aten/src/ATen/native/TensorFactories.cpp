#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

namespace at {
namespace native {

Tensor randn_like(const Tensor& self) {
  return self.type().randn(self.sizes());
}

}
}
