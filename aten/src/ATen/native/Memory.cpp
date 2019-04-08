#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NativeFunctions.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/util/Exception.h>

namespace at {
namespace native {

Tensor pin_memory(const Tensor& self) {
  if (self.type().backend() != Backend::CPU) {
    AT_ERROR("cannot pin '", self.type().toString(), "' only dense CPU tensors can be pinned");
  }
  auto* allocator = detail::getCUDAHooks().getPinnedMemoryAllocator();
  auto tensor = self.dispatch_type().tensorWithAllocator(self.sizes(), self.strides(), allocator);
  tensor.copy_(self);
  return tensor;
}

// Exposes at::has_internal_overlap as an operator for testing purposes
int64_t _debug_has_internal_overlap(const Tensor& self) {
  return static_cast<int64_t>(at::has_internal_overlap(self));
}

}
}
