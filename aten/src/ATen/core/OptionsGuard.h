#pragma once

#include <ATen/core/Device.h>
#include <ATen/core/Layout.h>
#include <ATen/core/ScalarType.h>
#include <ATen/core/TensorOptions.h>
#include <ATen/core/optional.h>
#include <ATen/core/Macros.h>

namespace at {

#if !AT_MOBILE

/// A wrapper over a thread local TensorOptions instance.
struct DefaultTensorOptions {
  /// Returns the current thread local default options.
  /// Defined in OptionsGuard.cpp because we can't use optional in headers, due
  /// to Windows and other compilers.
  /// TODO: The inability to use optional in headers is no longer true
  AT_API static TensorOptions& get();

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
///
/// You should NOT use OptionsGuard for internal code in ATen; it is reserved
/// for end users.
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

#else // AT_MOBILE

struct DefaultTensorOptions {
  AT_API static const TensorOptions& get();
private:
  static TensorOptions options_;
};

template<typename T = void>
struct OptionsGuard {
  OptionsGuard() {
      static_assert(!std::is_same<T, void>::value,
                    "OptionsGuard is not supported on mobile; please pass around TensorOptions manually");
  }
};

#endif

} // namespace at
