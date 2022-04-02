#include <ATen/core/PythonModeTLS.h>

namespace at { namespace impl {

thread_local std::shared_ptr<SafePyObject> pythonModeState;

void PythonModeTLS::set_state(std::shared_ptr<SafePyObject> state) {
  pythonModeState = std::move(state);
  if (state) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, true);
    c10::impl::tls_set_dispatch_key_included(DispatchKey::PythonTLSSnapshot, true);
  } else {
    PythonModeTLS::reset_state();
  }
}

const std::shared_ptr<SafePyObject>& PythonModeTLS::get_state() {
  return pythonModeState;
}

void PythonModeTLS::reset_state() {
  pythonModeState.reset();
  c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
  c10::impl::tls_set_dispatch_key_included(DispatchKey::PythonTLSSnapshot, false);
}

} // namespace impl
} // namespace at
