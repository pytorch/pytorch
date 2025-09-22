#include <ATen/core/PythonOpRegistrationTrampoline.h>
#include <c10/core/impl/PyInterpreterHooks.h>

namespace at::impl {

// Since torch/deploy and multipy are deprecated, we only support one Python
// interpreter per process. This simplifies to just using the global interpreter.

c10::impl::PyInterpreter* PythonOpRegistrationTrampoline::interpreter_ = nullptr;

c10::impl::PyInterpreter* PythonOpRegistrationTrampoline::getInterpreter() {
  // Simply delegate to the global PyInterpreter
  return c10::impl::getGlobalPyInterpreter();
}

bool PythonOpRegistrationTrampoline::registerInterpreter(
    c10::impl::PyInterpreter* interp) {
  // In single-interpreter mode, just track if we've already registered
  if (interpreter_ != nullptr) {
    // Already registered
    return false;
  }
  interpreter_ = interp;
  return true;
}

} // namespace at::impl
