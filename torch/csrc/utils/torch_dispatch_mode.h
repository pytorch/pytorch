#pragma once

#include <ATen/core/TorchDispatchModeTLS.h>

namespace torch {
namespace torch_dispatch_mode {

struct StashTorchDispatchModeGuard {
 public:
  StashTorchDispatchModeGuard(
      std::shared_ptr<at::SafePyObject> new_state = nullptr) {
    saved_ = at::impl::TorchDispatchModeTLS::get_state();
    at::impl::TorchDispatchModeTLS::set_state(std::move(new_state));
  }

  ~StashTorchDispatchModeGuard() {
    at::impl::TorchDispatchModeTLS::set_state(saved_);
  }

 private:
  std::shared_ptr<at::SafePyObject> saved_;
};

} // namespace torch_dispatch_mode
} // namespace torch
