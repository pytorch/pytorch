#include <ATen/core/PythonOpRegistrationTrampoline.h>
#include <c10/util/Logging.h>

namespace at::impl {

// Simplified implementation for single Python interpreter only
c10::impl::PyInterpreter*
    PythonOpRegistrationTrampoline::interpreter_{nullptr};

c10::impl::PyInterpreter* PythonOpRegistrationTrampoline::getInterpreter() {
  return interpreter_;
}

bool PythonOpRegistrationTrampoline::registerInterpreter(
    c10::impl::PyInterpreter* interp) {
  if (interpreter_ == nullptr) {
    interpreter_ = interp;
    return true;
  }
  // Already registered - this should not happen with single interpreter
  TORCH_WARN("Attempting to register Python interpreter when one is already registered. "
             "This should not happen with single interpreter support.");
  return false;
}

} // namespace at::impl
