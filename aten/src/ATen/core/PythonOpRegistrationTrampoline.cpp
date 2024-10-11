#include <ATen/core/PythonOpRegistrationTrampoline.h>

namespace at::impl {

// The strategy is that all python interpreters attempt to register themselves
// as the main interpreter, but only one wins.  Only that interpreter is
// allowed to interact with the C++ dispatcher.  Furthermore, when we execute
// logic on that interpreter, we do so hermetically, never setting pyobj field
// on Tensor.

std::atomic<c10::impl::PyInterpreter*>
    PythonOpRegistrationTrampoline::interpreter_{nullptr};

c10::impl::PyInterpreter* PythonOpRegistrationTrampoline::getInterpreter() {
  return PythonOpRegistrationTrampoline::interpreter_.load();
}

bool PythonOpRegistrationTrampoline::registerInterpreter(
    c10::impl::PyInterpreter* interp) {
  c10::impl::PyInterpreter* expected = nullptr;
  interpreter_.compare_exchange_strong(expected, interp);
  if (expected != nullptr) {
    // This is the second (or later) Python interpreter, which means we need
    // non-trivial hermetic PyObject TLS
    c10::impl::HermeticPyObjectTLS::init_state();
    return false;
  } else {
    return true;
  }
}

} // namespace at::impl
