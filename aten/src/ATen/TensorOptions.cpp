#include <ATen/TensorOptions.h>

#include <ATen/Context.h>
#include <ATen/Device.h>
#include <ATen/Layout.h>
#include <ATen/ScalarType.h>
#include <ATen/Tensor.h>
#include <ATen/optional.h>

namespace at {
TensorOptions::TensorOptions(Tensor tensor) : TensorOptions(tensor.type()) {
  this->device(Device(tensor));
}

TensorOptions::TensorOptions(const Type& type, optional<int32_t> device_index)
    : TensorOptions() {
  type_ = &type;
  this->dtype(type.scalarType());
  this->device({toDense(type.backend()), device_index});
  this->layout(layout_from_type(type));
}

TensorOptions& TensorOptions::device(Device device) {
  device_ = std::move(device);
  return *this;
}

TensorOptions& TensorOptions::device_index(int32_t device_index) {
  return device({Device::Type::CUDA, device_index});
}

TensorOptions& TensorOptions::dtype(ScalarType dtype) {
  dtype_ = dtype;
  return *this;
}

TensorOptions& TensorOptions::layout(Layout layout) {
  layout_ = layout;
  return *this;
}

const Device& TensorOptions::device() const noexcept {
  return device_;
}

const at::optional<int32_t>& TensorOptions::device_index() const noexcept {
  return device_.index();
}

ScalarType TensorOptions::dtype() const noexcept {
  return dtype_;
}

Layout TensorOptions::layout() const noexcept {
  return layout_;
}

const Type& TensorOptions::type() const {
  if (type_ != nullptr) {
    return *type_;
  }
  Backend backend;
  if (device_.type() == Device::Type::CPU) {
    backend = (layout_ == kStrided) ? kCPU : kSparseCPU;
  } else {
    backend = (layout_ == kStrided) ? kCUDA : kSparseCUDA;
  }
  return getType(backend, dtype_);
}

Tensor TensorOptions::apply(Tensor tensor) const {
  return tensor;
}
} // namespace at
