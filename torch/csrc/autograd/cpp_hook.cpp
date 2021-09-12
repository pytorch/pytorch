#include <torch/csrc/autograd/cpp_hook.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/custom_function.h>

namespace {
using torch::autograd::Variable;
void check_single_result (Variable value, Variable result, std::string hook_name) {
  if (!value.defined()) {
    throw std::runtime_error("can't replace a empty gradient with a non-empty value");
  }
  torch::autograd::check_variable_result(value, result, hook_name);
}
}

namespace torch { namespace autograd {

// NOLINTNEXTLINE(modernize-pass-by-value)
CppFunctionPreHook::CppFunctionPreHook(const std::shared_ptr<hooks_list> &hooks, int value_idx)
: hooks_(hooks)
, value_idx_(value_idx)
{}

variable_list CppFunctionPreHook::operator()(const variable_list& values) {
  auto value = values[value_idx_];
  for (unsigned i = 0; i < hooks_->size(); ++i) {
    auto &hook = (*hooks_)[i];
    if (!hook) {
      // hook was removed
      continue;
    }
    auto res = hook(value);
    if (!res.defined()) {
      // Don't change gradient
      continue;
    }
    check_single_result(value, res, c10::to_string(i));
    value = res;
  }
  variable_list results(values);
  results[value_idx_] = value;
  return results;
}

}} // namespace torch::autograd
