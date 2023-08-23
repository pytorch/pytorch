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

struct ProxyOrFakeModeGuard {
 public:
  ProxyOrFakeModeGuard(bool is_proxy) {
    if (is_proxy) {
      saved_mode_ = c10::impl::TorchDispatchModeTLS::unset_proxy_mode();
    } else {
      saved_mode_ = c10::impl::TorchDispatchModeTLS::unset_fake_mode();
    }
    is_proxy_ = is_proxy;
  }

  ~ProxyOrFakeModeGuard() {
    if (is_proxy_) {
      c10::impl::TorchDispatchModeTLS::set_proxy_mode(std::move(saved_mode_));
    } else {
      c10::impl::TorchDispatchModeTLS::set_fake_mode(std::move(saved_mode_));
    }
  }

  const std::shared_ptr<c10::SafePyObject>& get_cur_mode() {
    return saved_mode_;
  }

 private:
  std::shared_ptr<at::SafePyObject> saved_mode_;
  bool is_proxy_;
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
