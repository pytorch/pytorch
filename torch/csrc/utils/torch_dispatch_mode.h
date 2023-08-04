#pragma once

#include <c10/core/impl/TorchDispatchModeTLS.h>

namespace torch {
namespace torch_dispatch_mode {

struct StashTorchDispatchModeGuard {
 public:
  StashTorchDispatchModeGuard() {
    saved_mode_ = c10::impl::TorchDispatchModeTLS::pop_stack();
  }

  ~StashTorchDispatchModeGuard() {
    c10::impl::TorchDispatchModeTLS::push_onto_stack(std::move(saved_mode_));
  }

  const std::shared_ptr<c10::SafePyObject>& get_cur_mode() {
    return saved_mode_;
  }

 private:
  std::shared_ptr<at::SafePyObject> saved_mode_;
};

struct ProxyOrFakeOrFunctionalModeGuard {
 public:
  enum class GuardType { Proxy, Fake, Functional };

  ProxyOrFakeOrFunctionalModeGuard(GuardType guard_type) {
    if (guard_type == GuardType::Proxy) {
      saved_mode_ = c10::impl::TorchDispatchModeTLS::unset_proxy_mode();
    } else if (guard_type == GuardType::Fake) {
      saved_mode_ = c10::impl::TorchDispatchModeTLS::unset_fake_mode();
    } else {
      saved_mode_ = c10::impl::TorchDispatchModeTLS::unset_functional_mode();
    }
    guard_type_ = guard_type;
  }

  ~ProxyOrFakeOrFunctionalModeGuard() {
    if (guard_type_ == GuardType::Proxy) {
      c10::impl::TorchDispatchModeTLS::set_proxy_mode(std::move(saved_mode_));
    } else if (guard_type_ == GuardType::Fake) {
      c10::impl::TorchDispatchModeTLS::set_fake_mode(std::move(saved_mode_));
    } else {
      c10::impl::TorchDispatchModeTLS::set_functional_mode(
          std::move(saved_mode_));
    }
  }

  const std::shared_ptr<c10::SafePyObject>& get_cur_mode() {
    return saved_mode_;
  }

 private:
  std::shared_ptr<at::SafePyObject> saved_mode_;
  GuardType guard_type_;
};

struct StashTorchDispatchStackGuard {
 public:
  StashTorchDispatchStackGuard() {
    auto old = c10::impl::TorchDispatchModeTLS::get_state();
    c10::impl::TorchDispatchModeTLS::set_state(std::move(saved_state_));
    saved_state_ = std::move(old);
  }

  ~StashTorchDispatchStackGuard() {
    c10::impl::TorchDispatchModeTLS::set_state(std::move(saved_state_));
  }

 private:
  c10::impl::TorchDispatchModeTLS saved_state_;
};

} // namespace torch_dispatch_mode
} // namespace torch
