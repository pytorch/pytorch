#pragma once

#include <ATen/core/DefaultTensorOptions.h>
#include <ATen/core/TensorOptions.h>
#include <c10/macros/Macros.h>

namespace at {

/// Returns the current default options.
CAFFE2_API const DefaultTensorOptions& getDefaultTensorOptions();

#if !C10_MOBILE && !defined(CAFFE2_FB_LIMITED_MOBILE_CAPABILITY)

/// Get a mutable reference to the current thread local default options.
CAFFE2_API DefaultTensorOptions& mutateDefaultTensorOptions();

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
    mutateDefaultTensorOptions().merge(options);
  }

  /// Restores the original default options.
  ~OptionsGuard() {
    mutateDefaultTensorOptions() = original_;
  }

 private:
  /// The original options that were in place at the time of construction of
  /// this object.
  DefaultTensorOptions original_;
};

#else // C10_MOBILE

template<typename T = void>
struct OptionsGuard {
  OptionsGuard() {
      static_assert(!std::is_same<T, void>::value,
                    "OptionsGuard is not supported on mobile; please pass around TensorOptions manually");
  }
};

#endif

} // namespace at
