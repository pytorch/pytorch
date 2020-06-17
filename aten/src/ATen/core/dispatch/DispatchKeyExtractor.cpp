#include <ATen/core/dispatch/DispatchKeyExtractor.h>

#include <sstream>

namespace c10 {

void DispatchKeyExtractor::setOperatorHasKernelForBackend(DispatchKey k, bool has_kernel) {
  if (has_kernel) {
    operatorHasKernelForBackend_ = operatorHasKernelForBackend_.add(k);
  } else {
    operatorHasKernelForBackend_ = operatorHasKernelForBackend_.remove(k);
  }
}

void DispatchKeyExtractor::setOperatorHasFallthroughForBackend(DispatchKey k, bool has_fallthrough) {
  if (has_fallthrough) {
    operatorHasFallthroughForBackend_ = operatorHasFallthroughForBackend_.add(k);
  } else {
    operatorHasFallthroughForBackend_ = operatorHasFallthroughForBackend_.remove(k);
  }
}

std::string DispatchKeyExtractor::dumpState() const {
  std::ostringstream oss;
  for (size_t i=0; i < c10::utils::bitset::NUM_BITS(); ++i) {
    if (dispatch_arg_indices_reverse_.get(i)) {
      oss << "1";
    } else {
      oss << "0";
    }
  }
  oss << " " << operatorHasKernelForBackend_ << "\n";
  return oss.str();
}

void DispatchKeyExtractor::checkInvariants(const FunctionSchema& schema) const {
  TORCH_INTERNAL_ASSERT(makeBitsetForDispatchArgs(schema) == dispatch_arg_indices_reverse_);
}

} // namespace c10
