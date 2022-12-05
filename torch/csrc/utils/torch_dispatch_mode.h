#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>

namespace torch {
namespace torch_dispatch_mode {

void push_onto_dispatch_stack(std::shared_ptr<at::SafePyObject> mode);
std::shared_ptr<at::SafePyObject> pop_dispatch_stack();


struct StashTorchDispatchModeGuard {
 public:
  StashTorchDispatchModeGuard() {
    saved_mode_ = pop_dispatch_stack();
  }

  ~StashTorchDispatchModeGuard() {
    push_onto_dispatch_stack(std::move(saved_mode_));
  }

  const std::shared_ptr<c10::SafePyObject>& get_cur_mode() {
    return saved_mode_;
  }

 private:
  std::shared_ptr<at::SafePyObject> saved_mode_;
};

struct StashTorchDispatchStackGuard {
 public:
  StashTorchDispatchStackGuard() {
    const auto old = c10::impl::TorchDispatchModeTLS::get_state();
    c10::impl::TorchDispatchModeTLS::set_state(saved_state_);
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
