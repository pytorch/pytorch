#pragma once

#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/autograd/variable.h>

#include <ATen/Error.h>
#include <ATen/Tensor.h>
#include <ATen/TensorOptions.h>
#include <ATen/Type.h>

namespace torch {
struct TensorOptions : public at::TensorOptions {
  using at::TensorOptions::device;
  using at::TensorOptions::device_index;
  using at::TensorOptions::dtype;
  using at::TensorOptions::layout;
  using at::TensorOptions::TensorOptions;

  /// Sets the `requires_grad` property of the `TensorOptions`.
  TensorOptions& requires_grad(bool requires_grad) {
    requires_grad_ = requires_grad;
    return *this;
  }

  /// Returns the `requires_grad` property of the `TensorOptions`.
  bool requires_grad() const noexcept {
    return requires_grad_;
  }

  const at::Type& type() const override {
    return *autograd::VariableType::getType(at::TensorOptions::type());
  }

  /// Apply post-creation actions to a tensor, namely:
  /// - Set the `requires_grad` property,
  /// - Reset the version counter to zero.
  /// The latter is necessary because e.g. `at::ones(...)` will create a
  /// `Variable` inside and then mutate it via e.g. `fill_`. This means the
  /// `Variable` coming out of `at::ones` would already have a version counter
  /// of 1. To prevent this, we set it back to zero.
  at::Tensor apply(at::Tensor tensor) const override {
    AT_CHECK(
        at::isFloatingType(dtype_) || !requires_grad_,
        "Only Tensors of floating point dtype can require gradients");
    auto& variable = autograd::as_variable_ref(tensor);
    variable.set_version_counter(0);
    variable.set_requires_grad(requires_grad_);
    return tensor;
  }

  /// Override these methods to return `torch::TensorOptions` instead of
  /// `at::TensorOptions`.

  TensorOptions& device(at::Device device) override {
    at::TensorOptions::device(std::move(device));
    return *this;
  }

  TensorOptions& device_index(int32_t device_index) override {
    at::TensorOptions::device_index(device_index);
    return *this;
  }

  TensorOptions& dtype(at::ScalarType dtype) override {
    at::TensorOptions::dtype(dtype);
    return *this;
  }

  TensorOptions& layout(at::Layout layout) override {
    at::TensorOptions::layout(layout);
    return *this;
  }

 private:
  bool requires_grad_{false};
};

/// Convenience function that returns a `TensorOptions` object with the
/// `requires_grad` set to the given one.
inline TensorOptions requires_grad(bool requires_grad) {
  return TensorOptions().requires_grad(requires_grad);
}

/// These functions are like the ATen ones, but return `torch::TensorOptions`,
/// which leads to a `Variable` being created, while `at::TensorOptions` leads
/// to an `at::Tensor`.

inline TensorOptions dtype(at::ScalarType dtype) {
  return TensorOptions().dtype(dtype);
}

/// Convenience function that returns a `TensorOptions` object with the `layout`
/// set to the given one.
inline TensorOptions layout(at::Layout layout) {
  return TensorOptions().layout(layout);
}

/// Convenience function that returns a `TensorOptions` object with the `device`
/// set to the given one.
inline TensorOptions device(at::Device device) {
  return TensorOptions().device(std::move(device));
}

/// Convenience function that returns a `TensorOptions` object with the
/// `device_index` set to the given one.
inline TensorOptions device_index(int32_t device_index) {
  return TensorOptions().device_index(device_index);
}

} // namespace torch
