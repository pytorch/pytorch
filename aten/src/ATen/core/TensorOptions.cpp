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
  return device_.value_or(DefaultTensorOptions::get().device());
}

ScalarType TensorOptions::dtype() const noexcept {
  return dtype_.value_or(DefaultTensorOptions::get().dtype());
}

Layout TensorOptions::layout() const noexcept {
  return layout_.value_or(DefaultTensorOptions::get().layout());
}

bool TensorOptions::requires_grad() const noexcept {
  return requires_grad_.value_or(DefaultTensorOptions::get().requires_grad());
}

bool TensorOptions::is_variable() const noexcept {
  return is_variable_.value_or(DefaultTensorOptions::get().is_variable());
}

} // namespace at
