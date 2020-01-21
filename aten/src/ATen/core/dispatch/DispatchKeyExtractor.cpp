#include <ATen/core/dispatch/DispatchKeyExtractor.h>

namespace c10 {

void DispatchKeyExtractor::setIsOperatorOverridden(DispatchKey k, bool is_overridden) {
  if (is_overridden) {
    perOperatorOverriddenKernels_ = perOperatorOverriddenKernels_.add(k);
  } else {
    perOperatorOverriddenKernels_ = perOperatorOverriddenKernels_.remove(k);
  }
}

} // namespace c10
