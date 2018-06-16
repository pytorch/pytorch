#pragma once

#include <ATen/Context.h>
#include <ATen/Device.h>
#include <ATen/Layout.h>
#include <ATen/ScalarType.h>
#include <ATen/Tensor.h>
#include <ATen/Type.h>

#include <cstddef>
#include <utility>

namespace at {

/// A class to encapsulate construction axes of a `Tensor`.
/// `TensorOptions` is a virtual class to enable overriding of certain methods
/// by subclasses in other libraries, such as PyTorch. In PyTorch, there is a
/// `torch::TensorOptions` subclass of this `TensorOptions`, which changes
/// `type()` to return a variable type instead of a tensor type, such that
/// variables are created inside factory methods, instead of tensors.
struct TensorOptions {
  /// Constructs the `TensorOptions` with valid defaults, which are:
  /// - dtype: float
  /// - device: CPU
  /// - layout: strided
  /// - requires_grad: false
  TensorOptions() = default;

  /// Constructs the `TensorOptions` from the type of the given `Tensor`.
  /// If the `Tensor` has a CUDA type, the `device_index` will match that of the
  /// tensor. See the constructor from `Type` for the semantics w.r.t. the
  /// `type()` method.
  explicit TensorOptions(Tensor tensor, bool discard_runtime_type = false) {
    if (!discard_runtime_type) {
      type_ = &tensor.type();
    }
    this->dtype(tensor.dtype());
    this->device(tensor.device());
    this->layout(tensor.layout());
  }

  /// Constructs the `TensorOptions` from a type and a `device_index`.
  ///
  /// If `discard_runtime_type` is false (the default), the behavior of
  /// `TensorOptions::type()` is changed in that it will always return this
  /// `type`, irrespective of any `device` or `dtype` or `layout` specified at a
  /// later time. This is to ensure that when a `TensorOptions` object is
  /// constructed from a tensor's type, and that type has a dynamic type other
  /// than `at::Type` (e.g. `torch::autograd::VariableType`), constructing a new
  /// tensor from this `TensorOptions` will use this same derived type. If
  /// instead the given `type` were destructured into its components (backend,
  /// dtype and layout), information about the runtime type of the `Type` would
  /// be lost. Set `discard_runtime_type` to `true` to always destructure the
  /// type into its components and discard its runtime type.
  /* implicit */ TensorOptions(
      const Type& type,
      int32_t device_index = -1,
      bool discard_runtime_type = false) {
    if (!discard_runtime_type) {
      type_ = &type;
    }
    this->dtype(type.scalarType());
    this->device({type.backend(), device_index});
    this->layout(type.layout());
  }

  /// Constructs a `TensorOptions` object with the given layout.
  /* implicit */ TensorOptions(Layout layout) : TensorOptions() {
    this->layout(layout);
  }

  /// Constructs a `TensorOptions` object with the given device.
  /* implicit */ TensorOptions(Device device) : TensorOptions() {
    this->device(device);
  }

  /// Constructs a `TensorOptions` object from a backend, forwarded to the
  /// `Device` constructor.
  /* implicit */ TensorOptions(Backend backend)
      : TensorOptions(Device(backend)) {}

  /// Constructs a `TensorOptions` object with the given dtype.
  /* implicit */ TensorOptions(ScalarType dtype) : TensorOptions() {
    this->dtype(dtype);
  }

  /// Discards the runtime type stored if the `TensorOptions` was constructed
  /// from a `Tensor` or a `Type`. See the documentation of the constructor from
  /// a `Type` for implications on the behavior of the `type()` method on
  /// `TensorOptions`.
  const TensorOptions& discard_runtime_type() const {
    type_ = nullptr;
    return *this;
  }

  // NOTE: These methods are defined in TensorOptions.cpp because I get funny
  // linker errors for their missing definition if they're defined in the
  // header. Who knows why?

  /// Sets the device of the `TensorOptions`.
  TensorOptions& device(Device device) {
    device_ = std::move(device);
    return *this;
  }

  /// Sets the device of the `TensorOptions` to CUDA, and then sets the device
  /// index to the given one.
  TensorOptions& device_index(int32_t device_index) {
    return device({Device::Type::CUDA, device_index});
  }

  /// Sets the dtype of the `TensorOptions`.
  TensorOptions& dtype(ScalarType dtype) {
    dtype_ = dtype;
    return *this;
  }

  /// Sets the layout of the `TensorOptions`.
  TensorOptions& layout(Layout layout) {
    layout_ = layout;
    return *this;
  }

  /// Sets the `requires_grad` property of the `TensorOptions`.
  TensorOptions& requires_grad(bool requires_grad = true) {
    requires_grad_ = requires_grad;
    return *this;
  }

  /// Returns the device of the `TensorOptions`.
  const Device& device() const noexcept {
    return device_;
  }

  /// Returns the device index of the `TensorOptions`.
  int32_t device_index() const noexcept {
    return device_.index();
  }

  /// Returns the dtype of the `TensorOptions`.
  ScalarType dtype() const noexcept {
    return dtype_;
  }

  /// Returns the layout of the `TensorOptions`.
  Layout layout() const noexcept {
    return layout_;
  }

  /// Returns the `requires_grad` property of the `TensorOptions`.
  bool requires_grad() const noexcept {
    return requires_grad_;
  }

  /// Constructs an `at::Type` from the members of the `TensorOptions`.
  const Type& type() const {
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

 protected:
  ScalarType dtype_{kFloat};
  Device device_{Device::Type::CPU};
  Layout layout_{Layout::Strided};
  bool requires_grad_{false};
  // Not part of the observable API, so make `mutable` so we can set it to
  // `null` in `discard_runtime_type`.
  mutable const Type* type_{nullptr};
};

/// Convenience function that returns a `TensorOptions` object with the `dtype`
/// set to the given one.
inline TensorOptions dtype(ScalarType dtype) {
  return TensorOptions().dtype(dtype);
}

/// Convenience function that returns a `TensorOptions` object with the `layout`
/// set to the given one.
inline TensorOptions layout(Layout layout) {
  return TensorOptions().layout(layout);
}

/// Convenience function that returns a `TensorOptions` object with the `device`
/// set to the given one.
inline TensorOptions device(Device device) {
  return TensorOptions().device(std::move(device));
}

/// Convenience function that returns a `TensorOptions` object with the
/// `device_index` set to the given one.
inline TensorOptions device_index(int32_t device_index) {
  return TensorOptions().device_index(device_index);
}

/// Convenience function that returns a `TensorOptions` object with the
/// `requires_grad` set to the given one.
inline TensorOptions requires_grad(bool requires_grad = true) {
  return TensorOptions().requires_grad(requires_grad);
}
} // namespace at
