#include <c10/core/DispatchKeySet.h>
#include <c10/core/SafePyObject.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>

namespace c10 {
namespace impl {

thread_local TorchDispatchModeTLS torchDispatchModeState;

// MODE
void TorchDispatchModeTLS::set_mode(std::shared_ptr<SafePyObject> mode) {
  if (mode) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, true);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, true);
  } else {
    TorchDispatchModeTLS::reset_mode();
  }
  torchDispatchModeState.mode_ = std::move(mode);
}

const std::shared_ptr<SafePyObject>& TorchDispatchModeTLS::get_mode() {
  return torchDispatchModeState.mode_;
}

void TorchDispatchModeTLS::reset_mode() {
  torchDispatchModeState.mode_.reset();
  c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
  c10::impl::tls_set_dispatch_key_included(
      DispatchKey::PythonTLSSnapshot, false);
}

void TorchDispatchModeTLS::swap_mode(std::shared_ptr<SafePyObject>& mode) {
  if (mode) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, true);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, true);
  } else {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, false);
  }
  torchDispatchModeState.mode_.swap(mode);
}

// STACK
void TorchDispatchModeTLS::push_onto_stack(std::shared_ptr<SafePyObject> mode) {
  torchDispatchModeState.stack_.push_back(std::move(mode));
}

const std::shared_ptr<SafePyObject> TorchDispatchModeTLS::pop_stack() {
  TORCH_CHECK(
      torchDispatchModeState.stack_.size() > 0,
      "trying to pop from empty mode stack");
  const std::shared_ptr<SafePyObject> out =
      torchDispatchModeState.stack_.back();
  torchDispatchModeState.stack_.pop_back();
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
  return torchDispatchModeState.stack_.size();
}

// STATE

const TorchDispatchModeTLS& TorchDispatchModeTLS::get_state() {
  return torchDispatchModeState;
}

void TorchDispatchModeTLS::set_state(const TorchDispatchModeTLS& state) {
  torchDispatchModeState = state;
}

// UTIL

bool dispatch_mode_enabled() {
  return static_cast<bool>(c10::impl::TorchDispatchModeTLS::get_mode());
}

} // namespace impl
} // namespace c10
