#include <c10/util/irange.h>
#include <torch/csrc/autograd/cpp_hook.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/variable.h>

#include <utility>

namespace {
using torch::autograd::Variable;
void check_single_result(
    const at::TensorBase& value,
    const at::TensorBase& result,
    const std::string& hook_name) {
  if (!value.defined()) {
    throw std::runtime_error(
        "can't replace a empty gradient with a non-empty value");
  }
  torch::autograd::check_variable_result(value, result, hook_name);
}
} // namespace

namespace torch::autograd {

CppFunctionTensorPreHook::CppFunctionTensorPreHook(
    std::shared_ptr<hooks_list> hooks,
    size_t value_idx)
    : hooks_(std::move(hooks)), value_idx_(value_idx) {}

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
    check_single_result(value, res, std::to_string(i));
    value = std::move(res);
  }
  variable_list results(values);
  results[value_idx_] = value;
  return results;
}

CppFunctionSingleTensorPreHook::CppFunctionSingleTensorPreHook(
    std::function<at::TensorBase(const at::TensorBase&)> hook,
    size_t value_idx)
    : hook_(std::move(hook)), value_idx_(value_idx) {}

variable_list CppFunctionSingleTensorPreHook::operator()(
    const variable_list& values) {
  const auto& value = values[value_idx_];
  auto res = hook_(value);
  TORCH_INTERNAL_ASSERT(
      !res.defined(),
      "CppFunctionSingleTensorPreHook currently only supports hooks that don't return");
  variable_list results(values);
  return results;
}

} // namespace torch::autograd
