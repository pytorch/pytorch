#pragma once

#include <ATen/PythonTorchFunctionTLS.h>

namespace torch {
namespace overrides {

struct StashTorchFunctionModeGuard {
  StashTorchFunctionModeGuard() {
    cur_mode_ = at::impl::PythonTorchFunctionTLS::pop_stack();
  }
  ~StashTorchFunctionModeGuard() {
    at::impl::PythonTorchFunctionTLS::push_onto_stack(cur_mode_);
  }

  const std::shared_ptr<c10::SafePyObject>& get_cur_mode() {
    return cur_mode_;
  }

 private:
  std::shared_ptr<c10::SafePyObject> cur_mode_;
};

} // namespace overrides
} // namespace torch
