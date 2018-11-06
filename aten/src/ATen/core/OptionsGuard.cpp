#include <ATen/core/OptionsGuard.h>
#include <ATen/core/Layout.h>
#include "c10/util/Optional.h"

namespace at {

// In the CAFFE2_FB_LIMITED_MOBILE_CAPABILITY build setting,
// thread_local is not supported.  In that case, we don't provide
// an OptionsGuard; and force you to pass around options manually.
#if !C10_MOBILE && !defined(CAFFE2_FB_LIMITED_MOBILE_CAPABILITY)

DefaultTensorOptions& mutateDefaultTensorOptions() {
  static thread_local c10::optional<DefaultTensorOptions> options;
  /// This is an optional because of compiler bugs that mis-initialize static
  /// thread local variables. The workaround is lazy initialization, i.e.
  /// `getDefaultTensorOptions()` will initialize the `options` to a proper
  /// value upon first invocation.
  /// https://gcc.gnu.org/ml/gcc-bugs/2013-12/msg00026.html
  if (!options) {
    options.emplace();
  }
  return *options;
}

const DefaultTensorOptions& getDefaultTensorOptions() {
  return mutateDefaultTensorOptions();
}

#else

const DefaultTensorOptions& getDefaultTensorOptions() {
  static DefaultTensorOptions options;
  return options;
}

#endif

} // namespace at
