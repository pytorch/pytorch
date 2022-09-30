#include <c10/util/irange.h>
#include <torch/csrc/autograd/cpp_hook.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/variable.h>

namespace {
using torch::autograd::Variable;
void check_single_result(
    const at::TensorBase& value,
    const at::TensorBase& result,
    std::string hook_name) {
  if (!value.defined()) {
    throw std::runtime_error(
        "can't replace a empty gradient with a non-empty value");
  }
  torch::autograd::check_variable_result(value, result, hook_name);
}
} // namespace

namespace torch {
namespace autograd {

// NOLINTNEXTLINE(modernize-pass-by-value)
CppFunctionTensorPreHook::CppFunctionTensorPreHook(
    const std::shared_ptr<hooks_list>& hooks,
    int value_idx)
    : hooks_(hooks), value_idx_(value_idx) {}

variable_list CppFunctionTensorPreHook::operator()(
    const variable_list& values) {
  auto value = values[value_idx_];
  for (const auto i : c10::irange(hooks_->size())) {
    auto& hook = (*hooks_)[i];
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
    value = std::move(res);
  }
  variable_list results(values);
  results[value_idx_] = value;
  return results;
}

// NB: This is currently used in accumulate grad only, which cannot save
//     FunctionPreHooks on the node itself because the acc grad node won't be
//     kept alive until there's another node referencing it
// NOLINTNEXTLINE(modernize-pass-by-value)
CombinedFunctionPreHook::CombinedFunctionPreHook(
    std::vector<std::shared_ptr<FunctionPreHook>> hooks)
    : hooks_(hooks) {}

variable_list CombinedFunctionPreHook::operator()(const variable_list& values) {
  variable_list res = values;
  for (const auto& hook : hooks_) {
    res = (*hook)(res);
  }
  return res;
}

} // namespace autograd
} // namespace torch
