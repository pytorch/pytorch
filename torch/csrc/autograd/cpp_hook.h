#pragma once
#include <ATen/Tensor.h>
#include <torch/csrc/autograd/function_hook.h>
#include <functional>
#include <memory>

namespace torch { namespace autograd {

/// This struct exists so that AutogradMeta can simply forward-declare
/// CppHooksList
struct CppHooksList {
  std::vector<std::function<at::Tensor(const at::Tensor&)>> hooks_list_;
};

struct CppFunctionPreHook : public FunctionPreHook {
  CppFunctionPreHook(std::shared_ptr<CppHooksList> hooks, int value_idx);
  std::vector<at::Tensor> operator()(const std::vector<at::Tensor>& values) override;

  std::shared_ptr<CppHooksList> hooks_;
  int value_idx_;
};
}} // namespace torch::autograd
