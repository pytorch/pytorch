#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "c10/util/Optional.h"

namespace at {
namespace native {

// Since the given Device may not have device_index set (i.e., having it as -1
// representing the current device), we need to set the device_index before
// comparing against the current device object in Tensor.
// This always **copies** but this is intended because (1) we shouldn't modify
// input argument, and (2) Device is small anyways.
// NB: This ONLY works for CUDA device
static inline Device ensure_has_index(const Device &device) {
  if (!device.is_cuda() || device.has_index()) {
    return device;
  }
  return Device(device.type(), at::current_device());
}

static inline Tensor to_impl(const Tensor& self, const TensorOptions& options, bool non_blocking) {
  return self.type().toBackend(options.backend()).toScalarType(typeMetaToScalarType(options.dtype()))
                    .copy(self, non_blocking, options.device());
}

Tensor to(const Tensor& self, const TensorOptions& options, bool non_blocking, bool copy) {
  AT_CHECK(options.requires_grad_opt() == c10::nullopt,
           "to(options) expects unset requires_grad flag, but got "
           "options.requires_grad set as ", options.requires_grad());

  const auto & layout_opt = options.layout_opt();
  AT_CHECK(!layout_opt || self.layout() == layout_opt.value(),
           "to(options) doesn't support converting to a different layout, "
           "but got self.layout being ", self.layout(),
           " and options.layout set as ", options.layout());

  auto device_opt = options.device_opt();
  if (device_opt) {
    device_opt = ensure_has_index(device_opt.value());
  }
  const auto & dtype_opt = options.dtype_opt();
  if ((!device_opt || self.device() == device_opt.value()) &&
      (!dtype_opt  || self.dtype()  ==  dtype_opt.value()) && !copy) {
    return self;
  }
  auto specified_options = self.options();
  if (device_opt) {
    specified_options = specified_options.device(device_opt.value());
  }
  if (dtype_opt) {
    specified_options = specified_options.dtype(dtype_opt.value());
  }
  return to_impl(self, specified_options, non_blocking);
}

Tensor to(const Tensor& self, Device device, ScalarType dtype, bool non_blocking, bool copy) {
  device = ensure_has_index(device);
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

Tensor to(const Tensor& self, const Tensor& other, bool non_blocking, bool copy) {
  auto self_options = self.options();
  auto options = other.options();
  // Tensor.options() always have everything filled so we are happy and don't
  // even need to fill in device index.
  if (self_options == options && !copy) {
    return self;
  }
  return to_impl(self, options, non_blocking);
}

}} // namespace at::native
