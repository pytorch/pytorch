#pragma once

#include <ATen/core/Device.h>
#include <ATen/core/Layout.h>
#include <ATen/core/ScalarType.h>
#include <ATen/core/TensorOptions.h>
#include <ATen/core/optional.h>
#include <ATen/core/Macros.h>

namespace at {

/// Like TensorOptions, but all fields are guaranteed to be filled.
struct DefaultTensorOptions {
  ScalarType dtype()    const noexcept { return dtype_; }
  Device device()       const noexcept { return device_; }
  Layout layout()       const noexcept { return layout_; }
  bool requires_grad()  const noexcept { return requires_grad_; }
  bool is_variable()    const noexcept { return is_variable_; }

  DefaultTensorOptions& apply(const TensorOptions& options) {
    if (options.dtype_opt().has_value()) {
      dtype_ = options.dtype();
    }
    if (options.device_opt().has_value()) {
      device_ = options.device();
    }
    if (options.layout_opt().has_value()) {
      layout_ = options.layout();
    }
    if (options.requires_grad_opt().has_value()) {
      requires_grad_ = options.requires_grad();
    }
    if (options.is_variable_opt().has_value()) {
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
};

#if !AT_MOBILE && !defined(CAFFE2_FB_LIMITED_MOBILE_CAPABILITY)

/// Returns the current thread local default options.
CAFFE2_API DefaultTensorOptions& getDefaultTensorOptions();

/// RAII guard that stores the current default options upon construction, sets
/// the current default options to the ones given to its constructor, and
/// finally resets the options back to the original ones in the destructor.
///
/// You should NOT use OptionsGuard for internal code in ATen; it is reserved
/// for end users.
struct OptionsGuard {
  /// Stores the current default options and sets them to the given ones.
  explicit OptionsGuard(const TensorOptions& options)
      : original_(getDefaultTensorOptions()) { // copy
    getDefaultTensorOptions().apply(options);
  }

  /// Restores the original default options.
  ~OptionsGuard() {
    getDefaultTensorOptions() = original_;
  }

 private:
  /// The original options that were in place at the time of construction of
  /// this object.
  DefaultTensorOptions original_;
};

#else // AT_MOBILE

// Return the global, immutable default tensor options
CAFFE2_API const DefaultTensorOptions& getDefaultTensorOptions();

template<typename T = void>
struct OptionsGuard {
  OptionsGuard() {
      static_assert(!std::is_same<T, void>::value,
                    "OptionsGuard is not supported on mobile; please pass around TensorOptions manually");
  }
};

#endif

} // namespace at
