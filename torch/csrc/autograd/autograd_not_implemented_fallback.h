#pragma once

#include <torch/library.h>

namespace torch {
namespace autograd {

// DispatchKey::Autograd fallback for built-in operators
TORCH_API torch::CppFunction autogradNotImplementedFallback();

// DispatchKey::AdInplaceOrView fallback for built-in operators
TORCH_API torch::CppFunction autogradNotImplementedInplaceOrViewFallback();

// DispatchKey::Autograd fallback for all other operators (i.e. custom
// operators)
TORCH_API torch::CppFunction basicAutogradNotImplementedFallback();

enum class AutogradFallbackMode {
  Nothing, // Fallback is a redispatch
  Warn, // Fallback raises a warning if backward is called
  Error, // Fallback raises an error if backward is called
};

// Change the behavior of "basicAutogradNotImplementedFallback"
// In Python this is:
// - torch._C._set_autograd_fallback_mode(str) -> None
// - torch._C._get_autograd_fallback_mode() -> str
TORCH_API void setAutogradFallbackMode(AutogradFallbackMode mode);
TORCH_API AutogradFallbackMode getAutogradFallbackMode();

} // namespace autograd
} // namespace torch
