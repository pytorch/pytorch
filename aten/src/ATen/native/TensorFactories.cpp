#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

namespace at {
namespace native {

Tensor empty_like(const Tensor& self) {
  return self.type().tensor(self.sizes());
}

Tensor rand_like(const Tensor& self) {
  return self.type().rand(self.sizes());
}

Tensor randn_like(const Tensor& self) {
  return self.type().randn(self.sizes());
}

}
}
