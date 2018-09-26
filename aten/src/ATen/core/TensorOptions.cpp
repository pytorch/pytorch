#include <ATen/core/TensorOptions.h>

#include <ATen/core/Device.h>
#include <ATen/core/Layout.h>
#include <ATen/core/OptionsGuard.h>
#include <ATen/core/ScalarType.h>
#include <ATen/core/optional.h>
#include <ATen/core/ScalarType.h>

#include <iostream>

namespace at {

std::ostream& operator<<(
    std::ostream& stream,
    const TensorOptions& options) {
  return stream << "TensorOptions(dtype=" << options.dtype()
                << ", device=" << options.device()
                << ", layout=" << options.layout()
                << ", requires_grad=" << std::boolalpha
                << options.requires_grad() << ")";
}

Device TensorOptions::device() const noexcept {
  if (has_device_) {
    return device_;
  } else {
    return DefaultTensorOptions::get().device();
  }
}

ScalarType TensorOptions::dtype() const noexcept {
  if (has_dtype_) {
    return dtype_;
  } else {
    return DefaultTensorOptions::get().dtype();
  }
}

Layout TensorOptions::layout() const noexcept {
  if (has_layout_) {
    return layout_;
  } else {
    return DefaultTensorOptions::get().layout();
  }
}

bool TensorOptions::requires_grad() const noexcept {
  if (has_requires_grad_) {
    return requires_grad_;
  } else {
    return DefaultTensorOptions::get().requires_grad();
  }
}

bool TensorOptions::is_variable() const noexcept {
  if (has_is_variable_) {
    return is_variable_;
  } else {
    return DefaultTensorOptions::get().is_variable();
  }
}

} // namespace at
