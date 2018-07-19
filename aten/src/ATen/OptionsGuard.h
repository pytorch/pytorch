#pragma once

#include <ATen/Device.h>
#include <ATen/Layout.h>
#include <ATen/ScalarType.h>
#include <ATen/TensorOptions.h>
#include <ATen/optional.h>

namespace at {

/// A wrapper over a thread local TensorOptions instance.
struct DefaultTensorOptions {
  /// Returns the current thread local default options.
  /// Defined in OptionsGuard.cpp because we can't use optional in headers, due
  /// to Windows and other compilers.
  static TensorOptions& get();

 private:
  /// This is an optional because of compiler bugs that mis-initialize static
  /// thread local variables. The workaround is lazy initialization, i.e.
  /// `DefaultTensorOptions::get()` will initialize the `options_` to a proper
  /// value upon first invocation.
  /// https://gcc.gnu.org/ml/gcc-bugs/2013-12/msg00026.html
  static thread_local at::optional<TensorOptions> options_;
};

/// RAII guard that stores the current default options upon construction, sets
/// the current default options to the ones given to its constructor, and
/// finally resets the options back to the original ones in the destructor.
struct OptionsGuard {
  /// Stores the current default options and sets them to the given ones.
  explicit OptionsGuard(const TensorOptions& options)
      : original_(DefaultTensorOptions::get()) {
    DefaultTensorOptions::get() = options;
  }

  /// Restores the original default options.
  ~OptionsGuard() {
    DefaultTensorOptions::get() = original_;
  }

  /// Returns the original options that were in place at the time of
  /// construction of this object.
  const TensorOptions& original() {
    return original_;
  }

 private:
  /// The original options that were in place at the time of construction of
  /// this object.
  TensorOptions original_;
};

} // namespace at
