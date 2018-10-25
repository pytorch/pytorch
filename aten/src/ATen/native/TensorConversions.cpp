#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "c10/util/Optional.h"

namespace at {
namespace native {

static void ensure_has_index(Device* device) {
  if (!device->is_cuda() || device->has_index()) {
    return;
  }
  device->set_index(at::current_device());
}

static inline Tensor to_impl(const Tensor& self, const TensorOptions& options, bool non_blocking) {
  return self.type().toBackend(options.backend()).toScalarType(typeMetaToScalarType(options.dtype()))
                    .copy(self, non_blocking, options.device());
}

Tensor to(const Tensor& self, const TensorOptions& options, bool non_blocking, bool copy) {
  AT_CHECK(options.requires_grad_opt() == c10::nullopt,
           "to(options) expects unset requires_grad, but got "
           "options.requires_grad set as ", options.requires_grad());

  const auto & layout_opt = options.layout_opt();
  AT_CHECK(!layout_opt || self.layout() == layout_opt.value(),
           "to(options) doesn't support converting to a different layout, "
           "but got self.layout being ", self.layout(),
           " and options.layout set as ", options.layout());

  const auto & device_opt = options.device_opt();
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

Tensor to(const Tensor& self, const Tensor& other, bool non_blocking, bool copy) {
  auto self_options = self.options();
  auto options = other.options();
  if (self_options == options && !copy) {
    return self;
  }
  return to_impl(self, options, non_blocking);
}

}} // namespace at::native
