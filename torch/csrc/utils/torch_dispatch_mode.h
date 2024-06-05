#pragma once

#include <c10/core/impl/TorchDispatchModeTLS.h>

namespace torch::torch_dispatch_mode {

struct StashTorchDispatchModeGuard {
 public:
  StashTorchDispatchModeGuard() {
    if (c10::impl::TorchDispatchModeTLS::any_modes_set(
            /*skip_infra_modes=*/true)) {
      saved_mode_ = c10::impl::TorchDispatchModeTLS::pop_stack();
    } else {
      auto mode_and_key =
          c10::impl::TorchDispatchModeTLS::pop_highest_infra_mode();
      saved_mode_ = std::move(std::get<0>(mode_and_key));
      saved_mode_key_ = std::get<1>(mode_and_key);
    }
  }

  ~StashTorchDispatchModeGuard() {
    if (saved_mode_key_ != c10::nullopt) {
      c10::impl::TorchDispatchModeTLS::set_mode(
          saved_mode_, saved_mode_key_.value());
    } else {
      c10::impl::TorchDispatchModeTLS::push_non_infra_mode_onto_stack(
          std::move(saved_mode_));
    }
  }

  const std::shared_ptr<c10::impl::PyObject_TorchDispatchMode>& get_cur_mode() {
    return saved_mode_;
  }

 private:
  std::shared_ptr<c10::impl::PyObject_TorchDispatchMode> saved_mode_;
  std::optional<c10::impl::TorchDispatchModeKey> saved_mode_key_;
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

} // namespace torch::torch_dispatch_mode
