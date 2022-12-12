#pragma once

#include <c10/core/ModePyObjTrampoline.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <torch/csrc/utils/mode_utils.h>

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

  const std::shared_ptr<c10::ModePyObjTrampoline>& get_cur_mode() {
    return saved_mode_;
  }

 private:
  std::shared_ptr<c10::ModePyObjTrampoline> saved_mode_;
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
