#pragma once

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>

namespace torch { namespace autograd {

TORCH_API std::vector<Variable> _wrap_outputs(
  const std::unordered_set<at::TensorImpl*> &inputs,
  const std::unordered_set<at::TensorImpl*> &non_differentiable,
  const std::unordered_set<at::TensorImpl*> &dirty_inputs,
  const at::ArrayRef<Variable> raw_outputs,
  const std::shared_ptr<Function> &cdata);

}} // namespace torch::autograd
