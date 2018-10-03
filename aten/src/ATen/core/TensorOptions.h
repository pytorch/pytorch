#pragma once

#include <ATen/core/Backend.h>
#include <ATen/core/Device.h>
#include <ATen/core/Layout.h>
#include <ATen/core/ScalarType.h>

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
struct CAFFE2_API TensorOptions {
  // NB: Explicit construction of all optional fields is REQUIRED
  // to work around an nvcc bug.  Otherwise, you get:
  //
  //    Error: Internal Compiler Error (codegen): "there was an error in
  //    verifying the lgenfe output!"
  //
  // This bug only occurs when compiling with --expt-relaxed-constexpr
  TensorOptions() : dtype_(), device_(), layout_(), requires_grad_(), is_variable_() {}

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
    device_ = device;
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

  /// Sets the `is_variable` property on the `TensorOptions`.
  TensorOptions& is_variable(bool is_variable) {
    is_variable_ = is_variable;
    return *this;
  }

  /// Returns the device of the `TensorOptions`.
  Device device() const noexcept;

  /// Returns the device of the `TensorOptions`, or `nullopt` if
  /// device is not specified.
  optional<Device> device_opt() const noexcept { return device_; }

  /// Returns the device index of the `TensorOptions`.
  int32_t device_index() const noexcept {
    return device().index();
  }

  /// Returns the dtype of the `TensorOptions`.
  ScalarType dtype() const noexcept;

  /// Returns the dtype of the `TensorOptions`, or `nullopt` if
  /// device is not specified.
  optional<ScalarType> dtype_opt() const noexcept { return dtype_; }

  /// Returns the layout of the `TensorOptions`.
  Layout layout() const noexcept;

  /// Returns the layout of the `TensorOptions`, or `nullopt` if
  /// layout is not specified.
  optional<Layout> layout_opt() const noexcept { return layout_; }

  /// Returns the `requires_grad` property of the `TensorOptions`.
  bool requires_grad() const noexcept;

  /// Returns the `requires_grad` property of the `TensorOptions`, or `nullopt`
  /// if `requires_grad` is not specified.
  optional<bool> requires_grad_opt() const noexcept { return requires_grad_; }

  /// Returns the `is_variable` property of the `TensorOptions`.
  bool is_variable() const noexcept;

  /// Returns the `is_variable` property of the `TensorOptions`, or
  /// `nullopt` if `is_variable` is not specified.
  optional<bool> is_variable_opt() const noexcept { return is_variable_; }

  // Resolves the ATen backend specified by the current construction axes.
  Backend backend() const noexcept {
    Backend backend;
    if (device().type() == Device::Type::CPU) {
      backend = (layout() == kStrided) ? Backend::CPU : Backend::SparseCPU;
    } else {
      backend = (layout() == kStrided) ? Backend::CUDA : Backend::SparseCUDA;
    }
    return backend;
  }

 private:
  // WARNING: If you edit TensorOptions to add more options, you
  // must adjust the implementation of Tensor::options
  optional<ScalarType> dtype_;
  optional<Device> device_;
  optional<Layout> layout_;
  optional<bool> requires_grad_;
  optional<bool> is_variable_;
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

std::ostream& operator<<(
    std::ostream& stream,
    const TensorOptions& options);

} // namespace at
