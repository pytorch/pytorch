#include <ATen/core/dispatch/DispatchKeyExtractor.h>

namespace c10 {

void DispatchKeyExtractor::setOperatorHasKernelForBackend(DispatchKey k, bool has_kernel) {
  if (has_kernel) {
    operatorHasKernelForBackend_ = operatorHasKernelForBackend_.add(k);
  } else {
    operatorHasKernelForBackend_ = operatorHasKernelForBackend_.remove(k);
  }
}

} // namespace c10
