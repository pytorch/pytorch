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
    std::lock_guard<std::mutex> lock(mutex());
    options() = new_options;
  }

  /// Under a lock, copies the default `TensorOptions` object and returns this
  /// copy.
  static TensorOptions copy() {
    std::lock_guard<std::mutex> lock(mutex());
    auto local_copy = options();
    return local_copy;
  }

  /// Under a lock, assigns a `TensorOptions` object to the default instance,
  /// and returns the instance that was previously in place.
  static TensorOptions exchange(const TensorOptions& new_options) {
    std::lock_guard<std::mutex> lock(mutex());
    auto old_options = options();
    options() = new_options;
    return old_options;
  }

 private:
  static TensorOptions& options() {
    // Don't invoke the default constructor here, since `TensorOptions`'s
    // default constructor calls into `DefaultTensorOptions` itself!
    static TensorOptions options(
        kFloat, Device::Type::CPU, kStrided, /*requires_grad=*/false);
    return options;
  }

  static std::mutex& mutex() {
    static std::mutex mutex;
    return mutex;
  }
};
} // namespace at
