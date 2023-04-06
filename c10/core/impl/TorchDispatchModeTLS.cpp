#include <c10/core/DispatchKeySet.h>
#include <c10/core/SafePyObject.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>

#include <utility>

namespace c10 {
namespace impl {

thread_local TorchDispatchModeTLS torchDispatchModeState;

void TorchDispatchModeTLS::push_onto_stack(std::shared_ptr<SafePyObject> mode) {
  if (torchDispatchModeState.stack_.empty()) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, true);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, true);
  }
  torchDispatchModeState.stack_.push_back(std::move(mode));
}

const std::shared_ptr<SafePyObject> TorchDispatchModeTLS::pop_stack() {
  TORCH_CHECK(
      !torchDispatchModeState.stack_.empty(),
      "trying to pop from empty mode stack");
  std::shared_ptr<SafePyObject> out = torchDispatchModeState.stack_.back();
  torchDispatchModeState.stack_.pop_back();

  if (torchDispatchModeState.stack_.empty()) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, false);
  }
  return out;
}

const std::shared_ptr<SafePyObject>& TorchDispatchModeTLS::get_stack_at(
    int64_t idx) {
  TORCH_CHECK(
      idx < static_cast<int64_t>(torchDispatchModeState.stack_.size()),
      "Tried to get stack at idx that's too big");
  return torchDispatchModeState.stack_[idx];
}

int64_t TorchDispatchModeTLS::stack_len() {
  return static_cast<int64_t>(torchDispatchModeState.stack_.size());
}

const TorchDispatchModeTLS& TorchDispatchModeTLS::get_state() {
  return torchDispatchModeState;
}

void TorchDispatchModeTLS::set_state(TorchDispatchModeTLS state) {
  torchDispatchModeState = std::move(state);
  if (torchDispatchModeState.stack_.empty()) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, false);
  } else {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, true);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, true);
  }
}

// UTIL

bool dispatch_mode_enabled() {
  return !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Python) &&
      TorchDispatchModeTLS::stack_len() > 0;
}

} // namespace impl
} // namespace c10
