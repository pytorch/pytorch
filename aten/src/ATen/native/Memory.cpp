#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/util/Exception.h>
#include <c10/core/Storage.h>

namespace at {
namespace native {

namespace {
static int64_t inline get_device_index_with_primary_context() {
  const auto& cuda_hooks = detail::getCUDAHooks();
  // check current device first
  int64_t current_device_index = cuda_hooks.current_device();
  if (current_device_index >= 0) {
    if (cuda_hooks.hasPrimaryContext(current_device_index)) {
      return current_device_index;
    }
  }
  for (int64_t device_index = 0; device_index < cuda_hooks.getNumGPUs(); device_index++) {
    if (device_index == current_device_index) continue;
    if (cuda_hooks.hasPrimaryContext(device_index)) {
      return device_index;
    }
  }
  return -1;
}
}

Tensor pin_memory(const Tensor& self) {
  if (self.type().backend() != Backend::CPU) {
    AT_ERROR("cannot pin '", self.type().toString(), "' only dense CPU tensors can be pinned");
  }

  // Pinned memory pointers allocated by any device can be directly used by any
  // other device, regardless of the current device at the time of allocation,
  // since we assume unified addressing.
  // So we grab any existing primary context, if available.
  // See pytorch/pytorch#21081.
  at::OptionalDeviceGuard device_guard;
  auto primary_ctx_device_index = get_device_index_with_primary_context();
  if (primary_ctx_device_index >= 0) {
    device_guard.reset_device(at::Device(at::DeviceType::CUDA, primary_ctx_device_index));
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
