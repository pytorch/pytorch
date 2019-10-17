#pragma once
#include <torch/csrc/autograd/function_hook.h>
#include <functional>
#include <memory>

namespace torch { namespace autograd {

/// This struct exists so that AutogradMeta can simply forward-declare
/// CppHooksList
struct CppHooksList {
  std::vector<std::function<Variable(const Variable&)>> hooks_list_;
};

struct CppFunctionPreHook : public FunctionPreHook {
  CppFunctionPreHook(std::shared_ptr<CppHooksList> hooks, int value_idx);
  variable_list operator()(const variable_list& values) override;

  std::shared_ptr<CppHooksList> hooks_;
  int value_idx_;
};
}} // namespace torch::autograd
