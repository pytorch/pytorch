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

std::string DispatchKeyExtractor::dumpState() const {
  std::ostringstream oss;
  oss << num_args_ << " " << operatorHasKernelForBackend_ << "\n";
  return oss.str();
}

void DispatchKeyExtractor::checkInvariants(const FunctionSchema& schema) const {
  TORCH_INTERNAL_ASSERT(schema.arguments().size() == num_args_);
}

} // namespace c10
