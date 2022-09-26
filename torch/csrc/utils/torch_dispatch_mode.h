#pragma once

#include <c10/core/impl/TorchDispatchModeTLS.h>

namespace torch {
namespace torch_dispatch_mode {

struct StashTorchDispatchModeGuard {
 public:
  StashTorchDispatchModeGuard() {
    c10::impl::TorchDispatchModeTLS::swap_mode(saved_mode_);
  }

  ~StashTorchDispatchModeGuard() {
    c10::impl::TorchDispatchModeTLS::set_mode(std::move(saved_mode_));
  }

 private:
  std::shared_ptr<at::SafePyObject> saved_mode_;
};

} // namespace torch_dispatch_mode
} // namespace torch
