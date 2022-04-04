#pragma once

#include <ATen/PythonTorchFunctionTLS.h>

namespace torch {
namespace overrides {

struct no_torch_function_mode {
  no_torch_function_mode() { at::impl::PythonTorchFunctionTLS::swap_mode(old_mode_); }
  ~no_torch_function_mode() { at::impl::PythonTorchFunctionTLS::set_mode(std::move(old_mode_)); }
private:
  std::shared_ptr<c10::SafePyObject> old_mode_ = nullptr;
};

} // namespace overrides
} // namespace torch
