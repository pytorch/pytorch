#pragma once
#include <torch/csrc/autograd/function_hook.h>
#include <functional>
#include <memory>

namespace torch::autograd {

using hooks_list =
    std::vector<std::function<at::TensorBase(const at::TensorBase&)>>;

struct CppFunctionTensorPreHook : public FunctionPreHook {
  CppFunctionTensorPreHook(std::shared_ptr<hooks_list> hooks, size_t value_idx);
  variable_list operator()(const variable_list& values) override;

  std::shared_ptr<hooks_list> hooks_;
  size_t value_idx_;
};

struct CppFunctionSingleTensorPreHook : public FunctionPreHook {
  CppFunctionSingleTensorPreHook(
      std::function<at::TensorBase(const at::TensorBase&)> hook,
      size_t value_idx);
  variable_list operator()(const variable_list& values) override;

  std::function<at::TensorBase(const at::TensorBase&)> hook_;
  size_t value_idx_;
};

} // namespace torch::autograd
