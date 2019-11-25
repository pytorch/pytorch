#pragma once
#include <c10/util/ordered_dict.h>
#include <torch/csrc/autograd/function_hook.h>
#include <functional>
#include <memory>

namespace torch { namespace autograd {

using hooks_dict = c10::OrderedDict<unsigned, std::function<Variable(const Variable&)>>;

struct CppFunctionPreHook : public FunctionPreHook {
  CppFunctionPreHook(std::shared_ptr<hooks_dict> hooks, int value_idx);
  variable_list operator()(const variable_list& values) override;

  std::shared_ptr<hooks_dict> hooks_;
  int value_idx_;
};
}} // namespace torch::autograd
