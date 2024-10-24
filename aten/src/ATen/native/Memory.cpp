#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/Context.h>
#include <c10/core/Storage.h>
#include <ATen/EmptyTensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/CPUFunctions.h>
#else
#include <ATen/ops/_debug_has_internal_overlap_native.h>
#include <ATen/ops/_pin_memory.h>
#include <ATen/ops/is_pinned_native.h>
#include <ATen/ops/pin_memory_native.h>
#include <ATen/ops/_pin_memory_native.h>
#include <ATen/ops/empty_cpu_dispatch.h>
#endif

namespace at::native {

// Exposes at::has_internal_overlap as an operator for testing purposes
int64_t _debug_has_internal_overlap(const Tensor& self) {
  return static_cast<int64_t>(at::has_internal_overlap(self));
}

bool is_pinned(const Tensor& self, std::optional<c10::Device> device) {
  std::optional<c10::DeviceType> opt_device_type;
  if (device.has_value()) {
    TORCH_WARN_DEPRECATION(
        "The argument 'device' of Tensor.is_pinned() ",
        "is deprecated. Please do not pass this argument.")
    opt_device_type = device.value().type();
  } else {
    opt_device_type = at::getAccelerator();
  }

  if (!self.is_cpu() || // Only CPU tensors can be pinned
      !opt_device_type.has_value() || // there is no accelerator
      !at::isAccelerator(
          opt_device_type.value())) { // passed device not an accelerator
    return false;
  }

  return at::globalContext()
      .getAcceleratorHooksInterface(opt_device_type)
      .isPinnedPtr(self.storage().data());
}

Tensor pin_memory(const Tensor& self, std::optional<c10::Device> device) {
  if (device.has_value()) {
    TORCH_WARN_DEPRECATION(
        "The argument 'device' of Tensor.pin_memory() ",
        "is deprecated. Please do not pass this argument.")
  }
  // Kind of mad that I have to do two dynamic dispatches here, pretty
  // annoying
  if (self.is_pinned(device)) {
    return self;
  }
  return at::_pin_memory(self, device);
}

Tensor _pin_memory(const Tensor& self, std::optional<c10::Device> device) {
  TORCH_CHECK(self.device().is_cpu(), "cannot pin '", self.toString(), "' only dense CPU tensors can be pinned");

  std::optional<c10::DeviceType> opt_device_type;
  if (device.has_value()) {
    opt_device_type = device.value().type();
  }

  auto* allocator = at::globalContext()
                        .getAcceleratorHooksInterface(opt_device_type)
                        .getPinnedMemoryAllocator();
  auto storage = Storage(
      Storage::use_byte_size_t(),
      detail::computeStorageNbytes(
          self.sizes(), self.strides(), self.dtype().itemsize()),
      allocator,
      /*resizable=*/false);

  auto tensor = at::cpu::empty({0}, self.options()).set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
  return tensor;
}

} // namespace at::native
