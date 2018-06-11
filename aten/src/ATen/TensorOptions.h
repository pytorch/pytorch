#pragma once

#include <ATen/Device.h>
#include <ATen/Layout.h>
#include <ATen/ScalarType.h>
#include <ATen/optional.h>

#include <cstddef>

namespace at {
struct Type;
struct Tensor;
} // namespace at

namespace at {

/// A class to encapsulate construction axes of a `Tensor`.
/// `TensorOptions` is a virtual class to enable overriding of certain methods
/// by subclasses in other libraries, such as PyTorch. In PyTorch, there is a
/// `torch::TensorOptions` subclass of this `TensorOptions`, which changes
/// `type()` to return a variable type instead of a tensor type, such that
/// variables are created inside factory methods, instead of tensors.
/// Furthermore, it changes `apply` to perform certain post-creation steps, such
/// as setting the `requires_grad` property of a `Variable`.
struct TensorOptions {
  /// Constructs the `TensorOptions` with valid defaults, which are:
  /// - dtype: float
  /// - device: CPU
  /// - layout: strided
  TensorOptions() = default;

  /// Constructs the `TensorOptions` from the type of the given `Tensor`.
  /// If the `Tensor` has a CUDA type, the `device_index` will match that of the
  /// tensor. See the constructor from `Type` for the semantics w.r.t. the
  /// `type()` method.
  explicit TensorOptions(Tensor tensor);

  /// Constructs the `TensorOptions` from a type and optional `device_index`.
  ///
  /// NOTE: This changes the behavior of `TensorOptions::type()` in that it will
  /// always return this `type`, irrespective of any `device` or `dtype` or
  /// `layout` specified at a later time. This is to ensure that when a
  /// `TensorOptions` object is constructed from a tensor's type, and that type
  /// has a dynamic type other than `at::Type` (e.g.
  /// `torch::autograd::VariableType`), constructing a new tensor from this
  /// `TensorOptions` will use this same derived type. If instead the given
  /// `type` were destructured into its components (backend, dtype and layout),
  /// information about the runtime type of the `Type` would be lost.
  /* implicit */ TensorOptions(
      const Type& type,
      optional<int32_t> device_index = nullopt);

  /// Constructs a `TensorOptions` object with the given layout.
  /* implicit */ TensorOptions(Layout layout) : TensorOptions() {
    this->layout(layout);
  }

  /// Constructs a `TensorOptions` object with the given device.
  /* implicit */ TensorOptions(Device device) : TensorOptions() {
    this->device(device);
  }

  /// Constructs a `TensorOptions` object with the given dtype.
  /* implicit */ TensorOptions(ScalarType dtype) : TensorOptions() {
    this->dtype(dtype);
  }

  virtual ~TensorOptions() = default;

  // NOTE: These methods are defined in TensorOptions.cpp because I get funny
  // linker errors for their missing definition if they're defined in the
  // header. Who knows why?

  /// Sets the device of the `TensorOptions`.
  virtual TensorOptions& device(Device device);

  /// Sets the device of the `TensorOptions` to CUDA, and then sets the device
  /// index to the given one.
  virtual TensorOptions& device_index(int32_t device_index);

  /// Sets the dtype of the `TensorOptions`.
  virtual TensorOptions& dtype(ScalarType dtype);

  /// Sets the layout of the `TensorOptions`.
  virtual TensorOptions& layout(Layout layout);

  /// Returns the device of the `TensorOptions`.
  virtual const Device& device() const noexcept;

  /// Returns the optional device index of the `TensorOptions`.
  virtual const at::optional<int32_t>& device_index() const noexcept;

  /// Returns the dtype of the `TensorOptions`.
  virtual ScalarType dtype() const noexcept;

  /// Returns the layout of the `TensorOptions`.
  virtual Layout layout() const noexcept;

  /// Constructs an `at::Type` from the members of the `TensorOptions`.
  virtual const Type& type() const;

  /// Applies certain options to a `Tensor` post-creation.
  virtual Tensor apply(Tensor tensor) const;

 protected:
  ScalarType dtype_{kFloat};
  Device device_{kCPU};
  Layout layout_{Layout::Strided};
  const Type* type_{nullptr};
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
} // namespace at
