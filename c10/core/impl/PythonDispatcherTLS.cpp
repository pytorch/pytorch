#include <c10/core/DispatchKey.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/PythonDispatcherTLS.h>

namespace c10::impl {

thread_local static PyInterpreter* pythonDispatcherState;

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

PyInterpreter* PythonDispatcherTLS::get_state_for_pre_dispatch() {
  if (tls_is_dispatch_key_included(DispatchKey::PreDispatch) &&
      !tls_is_dispatch_key_excluded(DispatchKey::PreDispatch) &&
      !tls_is_dispatch_key_excluded(DispatchKey::Python) &&
      !tls_is_dispatch_key_excluded(DispatchKey::PythonDispatcher)) {
    return pythonDispatcherState;
  }
  return nullptr;
}

void PythonDispatcherTLS::reset_state() {
  pythonDispatcherState = nullptr;
  c10::impl::tls_set_dispatch_key_included(
      DispatchKey::PythonDispatcher, false);
}

} // namespace c10::impl
