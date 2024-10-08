#pragma once

#include <ATen/PythonTorchFunctionTLS.h>

namespace torch::overrides {

struct StashTorchFunctionModeGuard {
  StashTorchFunctionModeGuard() {
    cur_mode_ = at::impl::PythonTorchFunctionTLS::pop_stack();
  }
  ~StashTorchFunctionModeGuard() {
    at::impl::PythonTorchFunctionTLS::push_onto_stack(cur_mode_);
  }
  StashTorchFunctionModeGuard(const StashTorchFunctionModeGuard&) = delete;
  StashTorchFunctionModeGuard(StashTorchFunctionModeGuard&&) = delete;
  StashTorchFunctionModeGuard& operator=(const StashTorchFunctionModeGuard&) =
      delete;
  StashTorchFunctionModeGuard& operator=(StashTorchFunctionModeGuard&&) =
      delete;

  const std::shared_ptr<c10::SafePyObject>& get_cur_mode() {
    return cur_mode_;
  }

 private:
  std::shared_ptr<c10::SafePyObject> cur_mode_;
};

} // namespace torch::overrides
