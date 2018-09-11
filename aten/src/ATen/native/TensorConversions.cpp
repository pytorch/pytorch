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
  return self.to(self.options().device(device).dtype(dtype), non_blocking);
}

Tensor to(const Tensor& self, ScalarType dtype, bool non_blocking) {
  if (self.dtype() == dtype) {
    return self;
  }
  return self.to(self.options().dtype(dtype), non_blocking);
}

Tensor to(const Tensor& self, Device device, bool non_blocking) {
  ensure_has_index(&device);
  if (self.device() == device) {
    return self;
  }
  return self.to(self.options().device(device), non_blocking);
}

Tensor to(const Tensor& self, const Tensor& other, bool non_blocking) {
  return self.to(other.options());
}

}} // namespace at::native
