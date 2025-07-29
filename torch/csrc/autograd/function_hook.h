#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/Export.h>
#include <string>
#include <vector>

namespace torch::dynamo::autograd {
class CompiledNodeArgs;
class SwapSavedVariables;
struct PackedArgs;
} // namespace torch::dynamo::autograd

// A hook that's called on gradients

namespace torch::autograd {

using Variable = at::Tensor;
using variable_list = std::vector<Variable>;

struct TORCH_API FunctionPreHook {
  virtual ~FunctionPreHook() = default;
  virtual variable_list operator()(const variable_list& grads) = 0;
  // only implemented for python hooks, registers hook with compiled autograd
  virtual void compiled_args(
      torch::dynamo::autograd::CompiledNodeArgs& args) const {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
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
  virtual void compiled_args(
      torch::dynamo::autograd::CompiledNodeArgs& args) const {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        std::string("compiled_args nyi, see [Note: Compiled Autograd] ") +
            typeid(*this).name());
  }
};

struct TORCH_API PostAccumulateGradHook {
  virtual ~PostAccumulateGradHook() = default;
  virtual void operator()(const Variable& tensor) = 0;
  // only implemented for python hooks on nodes, registers hook with compiled
  // autograd
  virtual void compiled_args(
      torch::dynamo::autograd::CompiledNodeArgs& args) const {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        std::string("compiled_args nyi, see [Note: Compiled Autograd] ") +
            typeid(*this).name());
  }

  virtual void apply_with_saved(
      Variable&,
      torch::dynamo::autograd::SwapSavedVariables&) {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        std::string("compiled_args nyi, see [Note: Compiled Autograd] ") +
            typeid(*this).name());
  }
};

} // namespace torch::autograd
