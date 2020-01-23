#include <ATen/core/dispatch/DispatchKeyExtractor.h>

namespace c10 {

void DispatchKeyExtractor::setOperatorHasKernelForBackend(DispatchKey k, bool is_overridden) {
  if (is_overridden) {
    operatorHasKernelForBackend_ = operatorHasKernelForBackend_.add(k);
  } else {
    operatorHasKernelForBackend_ = operatorHasKernelForBackend_.remove(k);
  }
}

} // namespace c10
