#pragma once

#include <ATen/Device.h>
#include <ATen/Layout.h>
#include <ATen/ScalarType.h>
#include <ATen/TensorOptions.h>

#include <mutex>

namespace at {

/// A struct that provides thread safe access to a single `TensorOptions`
/// instance.
struct DefaultTensorOptions {
  /// Under a lock, assigns a `TensorOptions` object to the default instance.
  static void assign(const TensorOptions& new_options) {
    std::lock_guard<std::mutex> lock(mutex_);
    options_ = new_options;
  }

  /// Under a lock, copies the default `TensorOptions` object and returns this
  /// copy.
  static TensorOptions copy() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto local_copy = options_;
    return local_copy;
  }

  /// Under a lock, assigns a `TensorOptions` object to the default instance,
  /// and returns the instance that was previously in place.
  static TensorOptions exchange(const TensorOptions& new_options) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto old_options = options_;
    options_ = new_options;
    return old_options;
  }

 private:
  // Defined in TensorOptions.cpp.
  static TensorOptions options_;
  static std::mutex mutex_;
};
} // namespace at
