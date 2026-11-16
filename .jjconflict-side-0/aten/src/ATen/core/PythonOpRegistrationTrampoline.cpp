#include <ATen/core/PythonOpRegistrationTrampoline.h>

namespace at::impl {

std::atomic<c10::impl::PyInterpreter*>
    PythonOpRegistrationTrampoline::interpreter_{nullptr};

c10::impl::PyInterpreter* PythonOpRegistrationTrampoline::getInterpreter() {
  return PythonOpRegistrationTrampoline::interpreter_.load();
}

bool PythonOpRegistrationTrampoline::registerInterpreter(
    c10::impl::PyInterpreter* interp) {
  c10::impl::PyInterpreter* expected = nullptr;
  return interpreter_.compare_exchange_strong(expected, interp);
}

} // namespace at::impl
