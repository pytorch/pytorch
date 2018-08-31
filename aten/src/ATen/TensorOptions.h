#pragma once

#include <ATen/core/Backend.h>
#include <ATen/Context.h>
#include <ATen/core/Device.h>
#include <ATen/DeviceGuard.h>
#include <ATen/core/Layout.h>
#include <ATen/core/ScalarType.h>
#include <ATen/Tensor.h>
#include <ATen/Type.h>

#include <cstddef>
#include <iosfwd>
#include <utility>

namespace at {

/// A class to encapsulate construction axes of an Tensor.  TensorOptions was
/// designed to support the Python style API for specifying construction options
/// on factory functions, e.g.,
///
///     torch.zeros(2, 3, dtype=torch.int32)
///
/// Because C++ doesn't natively support keyword arguments, there must be
/// another way of specifying keyword-like arguments.  TensorOptions is a
/// builder class which can be used to construct this "dictionary" of keyword
/// arguments: functions which support TensorOptions conventionally take this
/// argument optionally as their last argument.
///
/// WARNING: In PyTorch, there are `torch::` variants of factory functions,
/// e.g., torch::zeros for at::zeros.  These return Variables (while the
/// stock ATen functions return plain Tensors).  If you mix these functions
/// up, you WILL BE SAD.
///
/// Rather than use the constructor of this class directly, you should prefer to
/// use the constructor functions, and then chain setter methods on top of them.
///
///     at::device(at::kCUDA).dtype(kInt)
///     at::dtype(at::kInt)
///
/// Additionally, anywhere a TensorOptions is expected, you can directly
/// pass at::kCUDA / at::kInt, and it will implicitly convert to a TensorOptions.
///
/// Here are some recommended ways to create a 2x2 tensor of zeros
/// with certain properties.  These all *implicitly* make use of
/// TensorOptions, even if they don't mention the class explicitly:
///
///     at::zeros({2,2}, at::kCUDA);
///     at::zeros({2,2}, at::kLong);
///     at::zeros({2,2}, at::device(at::kCUDA).dtype(at::kLong()));
///     at::zeros({2,2}, at::device({at::kCUDA, 1})); // place on device 1
///     at::zeros({2,2}, at::requires_grad());
///
struct AT_API TensorOptions {
  TensorOptions() : TensorOptions(/*use_thread_local_default_options=*/true) {}

  /// Constructs the `TensorOptions` with defaults taken from the thread local
  /// `TensorOptions` object if `use_thread_local_default_options`, else
  /// defaults to:
  /// - dtype: kFloat,
  /// - device: kCPU,
  /// - layout: kStrided,
  /// - requires_grad: false
  explicit TensorOptions(bool use_thread_local_default_options);

  /// Constructs the `TensorOptions` from the type of the given `Tensor`.
  /// If the `Tensor` has a CUDA type, the `device_index` will match that of the
  /// tensor. The `requires_grad` property of the tensor is ignored and set to
  /// false in the created `TensorOptions`.  See the constructor from `Type` for
  /// the semantics w.r.t. the `type()` method.
  explicit TensorOptions(Tensor tensor) {
    this->dtype(tensor.dtype());
    this->device(tensor.device());
    this->layout(tensor.layout());
    this->is_variable(tensor.is_variable());
  }

  /// Constructs the `TensorOptions` from a type and a `device_index`.
  /* implicit */ TensorOptions(
      const Type& type,
      int32_t device_index = -1) {
    this->dtype(type.scalarType());
    this->device({backendToDeviceType(type.backend()), device_index});
    this->layout(type.layout());
    this->is_variable(type.is_variable());
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
      : TensorOptions(Device(backendToDeviceType(backend))) {}

  /// Constructs a `TensorOptions` object from a device type, forwarded to the
  /// `Device` constructor.
  /* implicit */ TensorOptions(DeviceType device_type)
      : TensorOptions(Device(device_type)) {}

  /// Constructs a `TensorOptions` object with the given dtype.
  /* implicit */ TensorOptions(ScalarType dtype) : TensorOptions() {
    this->dtype(dtype);
  }

  /// True if all elements of the `TensorOptions` match that of the other.
  bool operator==(const TensorOptions& other) const noexcept {
    return dtype_ == other.dtype_ && layout_ == other.layout_ &&
        device_ == other.device_ && requires_grad_ == other.requires_grad_;
  }

  /// True if any of the elements of this `TensorOptions` do not match that of
  /// the other.
  bool operator!=(const TensorOptions& other) const noexcept {
    return !(*this == other);
  }

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
  TensorOptions& requires_grad(bool requires_grad) {
    requires_grad_ = requires_grad;
    return *this;
  }

  TensorOptions& is_variable(bool is_variable) {
    is_variable_ = is_variable;
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

  /// Returns the `is_variable` property of the `TensorOptions`.
  bool is_variable() const noexcept {
    return is_variable_;
  }

  /// Constructs an `at::Type` from the members of the `TensorOptions`.
  const Type& type() const {
    return at::globalContext().getMaybeVariableType(backend(), dtype_, is_variable_);
  }

 private:
  // Resolves the ATen backend specified by the current construction axes.
  Backend backend() const noexcept {
    Backend backend;
    if (device_.type() == Device::Type::CPU) {
      backend = (layout_ == kStrided) ? Backend::CPU : Backend::SparseCPU;
    } else {
      backend = (layout_ == kStrided) ? Backend::CUDA : Backend::SparseCUDA;
    }
    return backend;
  }

 private:
  ScalarType dtype_{kFloat};
  Device device_{Device::Type::CPU};
  Layout layout_{Layout::Strided};
  bool requires_grad_{false};
  bool is_variable_{false};
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
/// `device` set to CUDA and the `device_index` set to the given one.
inline TensorOptions device_index(int32_t device_index) {
  return TensorOptions().device_index(device_index);
}

/// Convenience function that returns a `TensorOptions` object with the
/// `requires_grad` set to the given one.
inline TensorOptions requires_grad(bool requires_grad = true) {
  return TensorOptions().requires_grad(requires_grad);
}

/// From Tensor.h
inline TensorOptions Tensor::options() const {
  return TensorOptions(*this);
}

namespace detail {
inline Tensor to(
    const Tensor& tensor,
    const TensorOptions& options,
    bool non_blocking) {
  // Don't copy if the options match.
  if (tensor.options() == options) {
    return tensor;
  }
  DeviceGuard guard(options.device());
  return options.type().copy(tensor, non_blocking);
}
} // namespace detail

inline Tensor Tensor::to(Device device, ScalarType dtype, bool non_blocking)
    const {
  if (this->device() == device && this->dtype() == dtype) {
    return *this;
  }
  return detail::to(*this, options().device(device).dtype(dtype), non_blocking);
}

inline Tensor Tensor::to(ScalarType dtype, bool non_blocking) const {
  if (this->dtype() == dtype) {
    return *this;
  }
  return detail::to(*this, options().dtype(dtype), non_blocking);
}

inline Tensor Tensor::to(Device device, bool non_blocking) const {
  if (this->device() == device) {
    return *this;
  }
  return detail::to(*this, options().device(device), non_blocking);
}
} // namespace at

std::ostream& operator<<(
    std::ostream& stream,
    const at::TensorOptions& options);
