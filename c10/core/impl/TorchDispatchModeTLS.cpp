#include <c10/core/DispatchKey.h>
#include <c10/core/SafePyObject.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>

#include <utility>

namespace c10 {
namespace impl {

thread_local TorchDispatchModeTLS torchDispatchModeState;

void TorchDispatchModeTLS::push_onto_stack(std::shared_ptr<SafePyObject> mode) {
  if (torchDispatchModeState.stack_.empty() &&
      torchDispatchModeState.proxy_mode_ == c10::nullopt &&
      torchDispatchModeState.fake_mode_ == c10::nullopt) {
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

  if (torchDispatchModeState.stack_.empty() &&
      torchDispatchModeState.proxy_mode_ == c10::nullopt &&
      torchDispatchModeState.fake_mode_ == c10::nullopt) {
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

const c10::optional<std::shared_ptr<SafePyObject>> TorchDispatchModeTLS::
    get_fake_mode() {
  return torchDispatchModeState.fake_mode_;
}
const c10::optional<std::shared_ptr<SafePyObject>> TorchDispatchModeTLS::
    get_proxy_mode() {
  return torchDispatchModeState.proxy_mode_;
}
void TorchDispatchModeTLS::set_fake_mode(std::shared_ptr<SafePyObject> mode) {
  TORCH_CHECK(
      torchDispatchModeState.fake_mode_ == c10::nullopt,
      "trying to set the current fake mode, but one already exists");
  if (torchDispatchModeState.stack_.empty() &&
      torchDispatchModeState.proxy_mode_ == c10::nullopt &&
      torchDispatchModeState.fake_mode_ == c10::nullopt) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, true);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, true);
  }
  torchDispatchModeState.fake_mode_ = mode;
}
void TorchDispatchModeTLS::set_proxy_mode(std::shared_ptr<SafePyObject> mode) {
  if (torchDispatchModeState.stack_.empty() &&
      torchDispatchModeState.proxy_mode_ == c10::nullopt &&
      torchDispatchModeState.fake_mode_ == c10::nullopt) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, true);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, true);
  }
  torchDispatchModeState.proxy_mode_ = std::move(mode);
}

const std::shared_ptr<SafePyObject> TorchDispatchModeTLS::unset_fake_mode() {
  TORCH_CHECK(
      torchDispatchModeState.fake_mode_ != c10::nullopt,
      "trying to unset the current fake mode, but there isn't one");
  auto out = *torchDispatchModeState.fake_mode_;
  torchDispatchModeState.fake_mode_ = c10::nullopt;
  if (torchDispatchModeState.stack_.empty() &&
      torchDispatchModeState.proxy_mode_ == c10::nullopt &&
      torchDispatchModeState.fake_mode_ == c10::nullopt) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, false);
  }
  return out;
}
const std::shared_ptr<SafePyObject> TorchDispatchModeTLS::unset_proxy_mode() {
  TORCH_CHECK(
      torchDispatchModeState.proxy_mode_ != c10::nullopt,
      "trying to call _unset_proxy_mode, but there currently is not one.");
  auto out = *torchDispatchModeState.proxy_mode_;
  torchDispatchModeState.proxy_mode_ = c10::nullopt;
  if (torchDispatchModeState.stack_.empty() &&
      torchDispatchModeState.proxy_mode_ == c10::nullopt &&
      torchDispatchModeState.fake_mode_ == c10::nullopt) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, false);
  }
  return out;
}

const c10::optional<std::shared_ptr<SafePyObject>> TorchDispatchModeTLS::
    maybe_highest_mode() {
  // First check for any modes on the mode stack
  const auto mode_stack_len = TorchDispatchModeTLS::stack_len();
  if (mode_stack_len > 0) {
    return get_stack_at(mode_stack_len - 1);
  }
  // Then check for a proxy mode if there is one
  if (torchDispatchModeState.proxy_mode_ != c10::nullopt) {
    return torchDispatchModeState.proxy_mode_;
  }
  // Finally, return a fake mode if there is one (or c10::nullopt)
  return torchDispatchModeState.fake_mode_;
}

const TorchDispatchModeTLS& TorchDispatchModeTLS::get_state() {
  return torchDispatchModeState;
}

void TorchDispatchModeTLS::set_state(TorchDispatchModeTLS state) {
  torchDispatchModeState = std::move(state);
  if (torchDispatchModeState.stack_.empty() &&
      torchDispatchModeState.proxy_mode_ == c10::nullopt &&
      torchDispatchModeState.fake_mode_ == c10::nullopt) {
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

bool dispatch_mode_enabled(bool skip_proxy_and_fake) {
  if (skip_proxy_and_fake) {
    return !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Python) &&
        TorchDispatchModeTLS::stack_len() > 0;
  }
  return !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Python) &&
      (TorchDispatchModeTLS::stack_len() > 0 ||
       TorchDispatchModeTLS::get_proxy_mode() != c10::nullopt ||
       TorchDispatchModeTLS::get_fake_mode() != c10::nullopt);
}

} // namespace impl
} // namespace c10
