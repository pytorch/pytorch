#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/Export.h>
#include <string>
#include <vector>

// A hook that's called on gradients

namespace torch {
namespace autograd {

class CompiledNodeArgs;
using Variable = at::Tensor;
using variable_list = std::vector<Variable>;

struct TORCH_API FunctionPreHook {
  virtual ~FunctionPreHook() = default;
  virtual variable_list operator()(const variable_list& grads) = 0;
  // only implemented for python hooks, registers hook with compiled autograd
  virtual void compiled_args(CompiledNodeArgs& args) {
    throw std::runtime_error(
        std::string("compiled_args nyi, see [Note: Compiled Autograd] ") +
        typeid(*this).name());
  }
};

struct TORCH_API FunctionPostHook {
  virtual ~FunctionPostHook() = default;
  virtual variable_list operator()(
      const variable_list& outputs /* grad_inputs */,
      const variable_list& inputs /* grad_outputs */) = 0;
  // only implemented for python hooks, registers hook with compiled autograd
  virtual void compiled_args(CompiledNodeArgs& args) {
    throw std::runtime_error(
        std::string("compiled_args nyi, see [Note: Compiled Autograd] ") +
        typeid(*this).name());
  }
};

} // namespace autograd
} // namespace torch
