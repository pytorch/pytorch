#pragma once

#include <ATen/PythonTorchFunctionTLS.h>

namespace torch {
namespace overrides {

// Corresponds to torch.overrides._no_torch_function_mode.  We discourage use
// of this in userland because it's non-compositional; there might be another
// mode waiting to go after you, and you shouldn't just blindly disable it.
// From C++ side, there is no such thing as compositional modes, there is one
// mode and of course you should be able to clear it.
struct no_torch_function_mode {
  no_torch_function_mode() { at::impl::PythonTorchFunctionTLS::swap_mode(old_mode_); }
  ~no_torch_function_mode() { at::impl::PythonTorchFunctionTLS::set_mode(std::move(old_mode_)); }
private:
  std::shared_ptr<c10::SafePyObject> old_mode_ = nullptr;
};

} // namespace overrides
} // namespace torch
