#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

namespace at {
namespace native {

static void ensure_has_index(Device* device) {
  if (!device->is_cuda() || device->has_index()) {
    return;
  }
  device->set_index(at::current_device());
}

static Tensor to_impl(const Tensor& self, const TensorOptions& options, bool non_blocking) {
  return self.type().toBackend(options.backend()).toScalarType(options.dtype())
                    .copy(self, non_blocking, options.device());
}

Tensor to(const Tensor& self, Device device, ScalarType dtype, bool non_blocking, bool copy) {
  ensure_has_index(&device);
  if (self.device() == device && self.dtype() == dtype && !copy) {
    return self;
  }
  return to_impl(self, self.options().device(device).dtype(dtype), non_blocking);
}

Tensor to(const Tensor& self, ScalarType dtype, bool non_blocking, bool copy) {
  if (self.dtype() == dtype && !copy) {
    return self;
  }
  return to_impl(self, self.options().dtype(dtype), non_blocking);
}

Tensor to(const Tensor& self, Device device, bool non_blocking, bool copy) {
  ensure_has_index(&device);
  if (self.device() == device && !copy) {
    return self;
  }
  return to_impl(self, self.options().device(device), non_blocking);
}

Tensor to(const Tensor& self, const Tensor& other, bool non_blocking, bool copy) {
  auto self_options = self.options();
  auto options = other.options();
  if (self_options == options && !copy) {
    return self;
  }
  return to_impl(self, options, non_blocking);
}

}} // namespace at::native
