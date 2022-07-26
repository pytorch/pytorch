#include <ATen/core/TorchDispatchModeTLS.h>
#include <c10/core/SafePyObject.h>
#include <c10/core/DispatchKeySet.h>

namespace at { namespace impl {

thread_local std::shared_ptr<SafePyObject> torchDispatchModeState;

void TorchDispatchModeTLS::set_state(std::shared_ptr<SafePyObject> state) {
  if (state) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, true);
    c10::impl::tls_set_dispatch_key_included(DispatchKey::PythonTLSSnapshot, true);
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
  c10::impl::tls_set_dispatch_key_included(DispatchKey::PythonTLSSnapshot, false);
}

bool dispatch_mode_enabled() {
  return static_cast<bool>(at::impl::TorchDispatchModeTLS::get_state());
}

bool tensor_has_dispatch(const at::Tensor& t) {
  DispatchKeySet key_set({DispatchKey::Python, DispatchKey::PythonTLSSnapshot});
  return t.key_set().has_any(key_set);
}

bool tensorlist_has_dispatch(at::ITensorListRef li) {
  for (const auto& t: li) {
    if (tensor_has_dispatch(t)) {
      return true;
    }
  }
  return false;
}

bool tensorlist_has_dispatch(at::IOptTensorListRef li) {
  for (const auto& t: li) {
    if (t && tensor_has_dispatch(*t)) {
      return true;
    }
  }
  return false;
}

} // namespace impl
} // namespace at
