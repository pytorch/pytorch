#pragma once

#include <c10/core/impl/TorchDispatchModeTLS.h>

namespace torch {
namespace torch_dispatch_mode {

struct StashTorchDispatchModeGuard {
 public:
  StashTorchDispatchModeGuard() {
    saved_mode_and_key_ = c10::impl::TorchDispatchModeTLS::pop_stack();
  }

  ~StashTorchDispatchModeGuard() {
    c10::impl::TorchDispatchModeTLS::push_onto_stack(
        std::move(std::get<0>(saved_mode_and_key_)),
        std::get<1>(saved_mode_and_key_));
  }

  const std::shared_ptr<c10::SafePyObject>& get_cur_mode() {
    return std::get<0>(saved_mode_and_key_);
  }

 private:
  std::tuple<
      std::shared_ptr<at::SafePyObject>,
      c10::optional<c10::impl::TorchDispatchModeKey>>
      saved_mode_and_key_;
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
