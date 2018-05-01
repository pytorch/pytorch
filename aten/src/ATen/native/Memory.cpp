#include "ATen/ATen.h"
#include "ATen/Error.h"
#include "ATen/NativeFunctions.h"
#include "ATen/PinnedMemoryAllocator.h"

namespace at {
namespace native {

Tensor pin_memory(const Tensor& self) {
  if (self.type().backend() != kCPU) {
    AT_ERROR("cannot pin '", self.type().toString(), "' only CPU memory can be pinned");
  }
  auto allocator = std::unique_ptr<Allocator>(new PinnedMemoryAllocator());
  auto tensor = self.type().tensorWithAllocator(self.sizes(), self.strides(), std::move(allocator));
  tensor.copy_(self);
  return tensor;
}

}
}
