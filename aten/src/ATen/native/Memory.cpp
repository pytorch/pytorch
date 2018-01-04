#include "ATen/ATen.h"
#include "ATen/PinnedMemoryAllocator.h"
#include "ATen/NativeFunctions.h"

namespace at {
namespace native {

Tensor pin_memory(const Tensor& self) {
  if (self.type().backend() != kCPU) {
    runtime_error("cannot pin '%s' only CPU memory can be pinned", self.type().toString());
  }
  auto allocator = std::unique_ptr<Allocator>(new PinnedMemoryAllocator());
  auto tensor = self.type().tensorWithAllocator(self.sizes(), self.strides(), std::move(allocator));
  tensor.copy_(self);
  return tensor;
}

}
}
