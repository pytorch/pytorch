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

Tensor to(const Tensor& self, Device device, ScalarType dtype, bool non_blocking) {
  ensure_has_index(&device);
  if (self.device() == device && self.dtype() == dtype) {
    return self;
  }
  return self.type().toScalarType(dtype).copy(self, non_blocking, device);
}

Tensor to(const Tensor& self, ScalarType dtype, bool non_blocking) {
  if (self.dtype() == dtype) {
    return self;
  }
  return self.type().toScalarType(dtype).copy(self, non_blocking, self.device());
}

Tensor to(const Tensor& self, Device device, bool non_blocking) {
  ensure_has_index(&device);
  if (self.device() == device) {
    return self;
  }
  return self.type().copy(self, non_blocking, device);
}

Tensor to(const Tensor& self, const Tensor& other, bool non_blocking) {
  auto self_options = self.options();
  auto options = other.options();
  if (self_options == options) {
    return self;
  }
  return self.type().toBackend(options.backend()).toScalarType(options.dtype())
                    .copy(self, non_blocking, options.device());
}

}} // namespace at::native
