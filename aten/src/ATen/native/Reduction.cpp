#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

namespace at {
namespace native {

// FixMe: remove once scalars are fully supported in PyTorch.
Tensor _scalar_sum(const Tensor& self) {
  return self.sum();
}

}
}
