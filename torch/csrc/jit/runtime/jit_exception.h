#pragma once

#include <stdexcept>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {

struct TORCH_API JITException : public std::runtime_error {
  explicit JITException(const std::string& msg);
};

} // namespace jit
} // namespace torch
