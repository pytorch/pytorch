#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/util/Exception.h>
#include <c10/core/Storage.h>

namespace at {
namespace native {

Tensor pin_memory(const Tensor& self) {
  if (self.type().backend() != Backend::CPU) {
    AT_ERROR("cannot pin '", self.type().toString(), "' only dense CPU tensors can be pinned");
  }
  at::OptionalDeviceGuard device_guard;
  // Pinned memory pointers allocated by any device can be directly used by any
  // other device, regardless of the current device at the time of allocation,
  // since we assume unified addressing.
  // So we grab any existing primary context, if available.
  // See pytorch/pytorch#21081.
  auto primary_ctx_device = detail::getCUDAHooks().getDeviceWithPrimaryContext();
  if (primary_ctx_device >= 0) {
    device_guard.reset_device(at::Device(at::DeviceType::CUDA, primary_ctx_device));
  }
  auto* allocator = detail::getCUDAHooks().getPinnedMemoryAllocator();
  auto storage = Storage(
      self.dtype(),
      detail::computeStorageSize(self.sizes(), self.strides()),
      allocator,
      /*resizable=*/false
  );
  auto tensor = at::empty({0}, self.options()).set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
  return tensor;
}

// Exposes at::has_internal_overlap as an operator for testing purposes
int64_t _debug_has_internal_overlap(const Tensor& self) {
  return static_cast<int64_t>(at::has_internal_overlap(self));
}

}
}
