#include <ATen/TensorOptions.h>

#include <ATen/Context.h>
#include <ATen/Device.h>
#include <ATen/Layout.h>
#include <ATen/ScalarType.h>
#include <ATen/Tensor.h>
#include <ATen/Type.h>
#include <ATen/optional.h>

namespace at {

TensorOptions::TensorOptions(Tensor tensor, bool discard_runtime_type)
    : TensorOptions(tensor.type(), discard_runtime_type) {
  this->device(Device(tensor));
}

TensorOptions::TensorOptions(
    const Type& type,
    at::optional<int32_t> device_index,
    bool discard_runtime_type) {
  if (!discard_runtime_type) {
    type_ = &type;
  }
  // We destructure the type anyway just so that you can access the individual
  // components like `layout()` or `dtype()`.
  this->dtype(type.scalarType());
  this->device({toDense(type.backend()), device_index});
  this->layout(layout_from_type(type));
}

TensorOptions::TensorOptions(const Type& type) : TensorOptions(type, nullopt) {}

TensorOptions::TensorOptions(Backend backend)
    : TensorOptions(Device(backend)) {}

TensorOptions::TensorOptions(Device device) : TensorOptions() {
  this->device(device);
}

TensorOptions& TensorOptions::device(Device device) {
  device_ = std::move(device);
  return *this;
}

TensorOptions& TensorOptions::device_index(int32_t device_index) {
  return device({Device::Type::CUDA, device_index});
}

} // namespace at
