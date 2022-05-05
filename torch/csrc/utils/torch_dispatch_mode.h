#pragma once

#include <ATen/core/TorchDispatchModeTLS.h>

namespace torch {
namespace torch_dispatch_mode {

struct StashTorchDispatchModeGuard {
public:
  StashTorchDispatchModeGuard() {
    saved_ = at::impl::TorchDispatchModeTLS::get_state();
    at::impl::TorchDispatchModeTLS::set_state(nullptr);
  }

  ~StashTorchDispatchModeGuard() {
    at::impl::TorchDispatchModeTLS::set_state(saved_);
  }
private:
  std::shared_ptr<at::SafePyObject> saved_;
};

} // namespace torch_dispatch_mode
} // namespace torch
