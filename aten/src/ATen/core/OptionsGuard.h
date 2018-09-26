#pragma once

#include <ATen/core/Device.h>
#include <ATen/core/Layout.h>
#include <ATen/core/ScalarType.h>
#include <ATen/core/TensorOptions.h>
#include <ATen/core/optional.h>
#include <ATen/core/Macros.h>

namespace at {

/// A wrapper over a thread local TensorOptions instance.
/// INVARIANT: all fields are NOT nullopt
struct DefaultTensorOptions {
  /// Returns the current thread local default options.
  /// Defined in OptionsGuard.cpp because we can't use optional in headers, due
  /// to Windows and other compilers.
  /// TODO: The inability to use optional in headers is no longer true
  CAFFE2_API static DefaultTensorOptions& get();

  DefaultTensorOptions() {}
  DefaultTensorOptions(const DefaultTensorOptions&) = default;
  DefaultTensorOptions& operator=(const DefaultTensorOptions&) = default;
  DefaultTensorOptions(DefaultTensorOptions&&) = default;
  DefaultTensorOptions& operator=(DefaultTensorOptions&&) = default;

  ScalarType dtype()    const { return dtype_; }
  Device device()       const { return device_; }
  Layout layout()       const { return layout_; }
  bool requires_grad()  const { return requires_grad_; }
  bool is_variable()    const { return is_variable_; }

  DefaultTensorOptions& apply(const TensorOptions& options) {
    if (options.has_dtype()) {
      dtype_ = options.dtype();
    }
    if (options.has_device()) {
      device_ = options.device();
    }
    if (options.has_layout()) {
      layout_ = options.layout();
    }
    if (options.has_requires_grad()) {
      requires_grad_ = options.requires_grad();
    }
    if (options.has_is_variable()) {
      is_variable_ = options.is_variable();
    }
    return *this;
  }

 private:
  ScalarType dtype_ = at::kFloat;
  Device device_ = at::kCPU;
  Layout layout_ = at::kStrided;
  bool requires_grad_ = false;
  bool is_variable_ = false;

// In the CAFFE2_FB_LIMITED_MOBILE_CAPABILITY build setting,
// thread_local is not supported.  In that case, we don't provide
// an OptionsGuard; and force you to pass around options manually.
#if !AT_MOBILE && !defined(CAFFE2_FB_LIMITED_MOBILE_CAPABILITY)

  /// This is an optional because of compiler bugs that mis-initialize static
  /// thread local variables. The workaround is lazy initialization, i.e.
  /// `DefaultTensorOptions::get()` will initialize the `options_` to a proper
  /// value upon first invocation.
  /// https://gcc.gnu.org/ml/gcc-bugs/2013-12/msg00026.html
  static thread_local at::optional<DefaultTensorOptions> options_;

#endif
};

#if !AT_MOBILE && !defined(CAFFE2_FB_LIMITED_MOBILE_CAPABILITY)

/// RAII guard that stores the current default options upon construction, sets
/// the current default options to the ones given to its constructor, and
/// finally resets the options back to the original ones in the destructor.
///
/// You should NOT use OptionsGuard for internal code in ATen; it is reserved
/// for end users.
struct OptionsGuard {
  /// Stores the current default options and sets them to the given ones.
  explicit OptionsGuard(const TensorOptions& options)
      : original_(DefaultTensorOptions::get()) { // copy
    DefaultTensorOptions::get().apply(options);
  }

  /// Restores the original default options.
  ~OptionsGuard() {
    DefaultTensorOptions::get() = original_;
  }

 private:
  /// The original options that were in place at the time of construction of
  /// this object.
  DefaultTensorOptions original_;
};

#else // AT_MOBILE

template<typename T = void>
struct OptionsGuard {
  OptionsGuard() {
      static_assert(!std::is_same<T, void>::value,
                    "OptionsGuard is not supported on mobile; please pass around TensorOptions manually");
  }
};

#endif

} // namespace at
