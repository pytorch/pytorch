#include <c10/core/DispatchKeySet.h>
#include <c10/core/SafePyObject.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/PythonDispatcherTLS.h>

namespace c10 {
namespace impl {

thread_local PyInterpreter* pythonDispatcherState;

void PythonDispatcherTLS::set_state(PyInterpreter* state) {
  if (state) {
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonDispatcher, true);
  } else {
    PythonDispatcherTLS::reset_state();
  }
  pythonDispatcherState = state;
}

PyInterpreter* PythonDispatcherTLS::get_state() {
  return pythonDispatcherState;
}

void PythonDispatcherTLS::reset_state() {
  pythonDispatcherState = nullptr;
  c10::impl::tls_set_dispatch_key_included(
      DispatchKey::PythonDispatcher, false);
}

} // namespace impl
} // namespace c10
