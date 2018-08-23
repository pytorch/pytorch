#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/core/Error.h"
#include "ATen/detail/CUDAHooksInterface.h"

namespace at {
namespace native {

Tensor pin_memory(const Tensor& self) {
  if (self.type().backend() != Backend::CPU) {
    AT_ERROR("cannot pin '", self.type().toString(), "' only CPU memory can be pinned");
  }
  auto* allocator = detail::getCUDAHooks().getPinnedMemoryAllocator();
  auto tensor = self.type().tensorWithAllocator(self.sizes(), self.strides(), allocator);
  tensor.copy_(self);
  return tensor;
}

}
}
