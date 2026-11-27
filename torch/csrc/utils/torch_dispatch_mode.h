#pragma once

#include <c10/core/impl/FakeTensorModeTLS.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
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
    if (saved_mode_key_.has_value()) {
      c10::impl::TorchDispatchModeTLS::set_mode(
          saved_mode_, saved_mode_key_.value());
    } else {
      c10::impl::TorchDispatchModeTLS::push_non_infra_mode_onto_stack(
          std::move(saved_mode_));
    }
  }
  StashTorchDispatchModeGuard(const StashTorchDispatchModeGuard&) = delete;
  StashTorchDispatchModeGuard(StashTorchDispatchModeGuard&&) = delete;
  StashTorchDispatchModeGuard& operator=(const StashTorchDispatchModeGuard&) =
      delete;
  StashTorchDispatchModeGuard& operator=(StashTorchDispatchModeGuard&&) =
      delete;

  const std::shared_ptr<c10::impl::PyObject_TorchDispatchMode>& get_cur_mode() {
    return saved_mode_;
  }

 private:
  std::shared_ptr<c10::impl::PyObject_TorchDispatchMode> saved_mode_;
  std::optional<c10::impl::TorchDispatchModeKey> saved_mode_key_;
};

struct StashTorchDispatchStackGuard {
 public:
  StashTorchDispatchStackGuard()
      : saved_fake_mode_(c10::impl::FakeTensorModeTLS::get_state()),
        saved_fake_included_(
            c10::impl::tls_is_dispatch_key_included(c10::DispatchKey::Fake)) {
    c10::impl::FakeTensorModeTLS::reset_state();
    auto old = c10::impl::TorchDispatchModeTLS::get_state();
    c10::impl::TorchDispatchModeTLS::set_state(std::move(saved_state_));
    saved_state_ = std::move(old);
  }
  StashTorchDispatchStackGuard(const StashTorchDispatchStackGuard&) = delete;
  StashTorchDispatchStackGuard(StashTorchDispatchStackGuard&&) = delete;
  StashTorchDispatchStackGuard& operator=(const StashTorchDispatchStackGuard&) =
      delete;
  StashTorchDispatchStackGuard& operator=(StashTorchDispatchStackGuard&&) =
      delete;

  ~StashTorchDispatchStackGuard() {
    c10::impl::TorchDispatchModeTLS::set_state(std::move(saved_state_));
    auto old_fake_mode =
        c10::impl::FakeTensorModeTLS::create_state(std::move(saved_fake_mode_));
    c10::impl::tls_set_dispatch_key_included(
        c10::DispatchKey::Fake, saved_fake_included_);
  }

 private:
  c10::impl::TorchDispatchModeTLS saved_state_;
  std::shared_ptr<c10::FakeTensorMode> saved_fake_mode_;
  bool saved_fake_included_;
};

} // namespace torch::torch_dispatch_mode
