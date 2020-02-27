
copy: fbcode/caffe2/torch/csrc/jit/runtime/jit_exception.h
copyrev: 8b49cbcc8447cf73ba8af191ac0cf4e0dfd58b29

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
