#include <ATen/core/PythonOpRegistrationTrampoline.h>
#include <c10/core/impl/PyInterpreterHooks.h>

// TODO: delete this
namespace at::impl {

c10::impl::PyInterpreter* PythonOpRegistrationTrampoline::interpreter_ = nullptr;

c10::impl::PyInterpreter* PythonOpRegistrationTrampoline::getInterpreter() {
  return c10::impl::getGlobalPyInterpreter();
}

bool PythonOpRegistrationTrampoline::registerInterpreter(
    c10::impl::PyInterpreter* interp) {
  if (interpreter_ != nullptr) {
    return false;
  }
  interpreter_ = interp;
  return true;
}

} // namespace at::impl
