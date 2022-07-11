#pragma once
#include <torch/csrc/autograd/function_hook.h>
#include <functional>
#include <memory>

namespace torch {
namespace autograd {

using hooks_list =
    std::vector<std::function<at::TensorBase(const at::TensorBase&)>>;

struct CppFunctionPreHook : public FunctionPreHook {
  CppFunctionPreHook(const std::shared_ptr<hooks_list>& hooks, int value_idx);
  variable_list operator()(const variable_list& values) override;

  std::shared_ptr<hooks_list> hooks_;
  int value_idx_;
};
} // namespace autograd
} // namespace torch
