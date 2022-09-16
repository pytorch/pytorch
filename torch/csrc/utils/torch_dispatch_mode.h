#pragma once

#include <c10/core/impl/TorchDispatchModeTLS.h>

namespace torch {
namespace torch_dispatch_mode {

struct StashTorchDispatchModeGuard {
 public:
  StashTorchDispatchModeGuard() {
    saved_ = c10::impl::TorchDispatchModeTLS::get_state();
    c10::impl::TorchDispatchModeTLS::set_state(nullptr);
  }

  ~StashTorchDispatchModeGuard() {
    c10::impl::TorchDispatchModeTLS::set_state(saved_);
  }

 private:
  std::shared_ptr<at::SafePyObject> saved_;
};

} // namespace torch_dispatch_mode
} // namespace torch
