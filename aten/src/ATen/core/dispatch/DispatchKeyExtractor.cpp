#include <ATen/core/dispatch/DispatchKeyExtractor.h>

namespace c10 {

void DispatchKeyExtractor::setIsFallthroughKernel(DispatchKey k, bool is_fallthrough) {
  if (is_fallthrough) {
    nonFallthroughKernels_ = nonFallthroughKernels_.remove(k);
  } else {
    nonFallthroughKernels_ = nonFallthroughKernels_.add(k);
  }
}

} // namespace c10
