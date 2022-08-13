#include <c10/core/DispatchKeySet.h>
#include <c10/core/SafePyObject.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>

namespace c10 {
namespace impl {

thread_local std::shared_ptr<SafePyObject> torchDispatchModeState;

void TorchDispatchModeTLS::set_state(std::shared_ptr<SafePyObject> state) {
  if (state) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, true);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, true);
  } else {
    TorchDispatchModeTLS::reset_state();
  }
  torchDispatchModeState = std::move(state);
}

const std::shared_ptr<SafePyObject>& TorchDispatchModeTLS::get_state() {
  return torchDispatchModeState;
}

void TorchDispatchModeTLS::reset_state() {
  torchDispatchModeState.reset();
  c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
  c10::impl::tls_set_dispatch_key_included(
      DispatchKey::PythonTLSSnapshot, false);
}

bool dispatch_mode_enabled() {
  return static_cast<bool>(c10::impl::TorchDispatchModeTLS::get_state());
}

} // namespace impl
} // namespace c10
