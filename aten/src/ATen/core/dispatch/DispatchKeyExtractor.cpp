#include <ATen/core/dispatch/DispatchKeyExtractor.h>

#include <sstream>

namespace c10 {

void DispatchKeyExtractor::setOperatorHasFallthroughForKey(DispatchKey k, bool has_fallthrough) {
  if (has_fallthrough) {
    nonFallthroughKeys_ = nonFallthroughKeys_.remove(k);
  } else {
    nonFallthroughKeys_ = nonFallthroughKeys_.add(k);
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
  oss << " " << nonFallthroughKeys_ << "\n";
  return oss.str();
}

void DispatchKeyExtractor::checkInvariants(const FunctionSchema& schema) const {
  TORCH_INTERNAL_ASSERT(makeBitsetForDispatchArgs(schema) == dispatch_arg_indices_reverse_);
}

} // namespace c10
