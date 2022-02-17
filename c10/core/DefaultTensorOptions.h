#pragma once

#include <c10/core/Backend.h>
#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/core/ScalarType.h>

namespace c10 {

struct TensorOptions;

/// Like TensorOptions, but all fields are guaranteed to be filled.
struct DefaultTensorOptions {
  DefaultTensorOptions() = default;

  caffe2::TypeMeta dtype() const noexcept {
    return dtype_;
  }
  Device device() const noexcept {
    return device_;
  }
  Layout layout() const noexcept {
    return layout_;
  }
  bool requires_grad() const noexcept {
    return requires_grad_;
  }

  // Defined in TensorOptions.h
  inline DefaultTensorOptions& merge(const TensorOptions& options);

 private:
  caffe2::TypeMeta dtype_ = caffe2::TypeMeta::Make<float>(); // 64-bit
  Device device_ = at::kCPU; // 32-bit
  Layout layout_ = at::kStrided; // 8-bit
  bool requires_grad_ = false; // 8-bit
};

inline const DefaultTensorOptions& getDefaultTensorOptions() {
  static const auto options = DefaultTensorOptions();
  return options;
}

} // namespace c10
