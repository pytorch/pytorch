#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/EmptyTensor.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/InferSize.h>
#include <ATen/TensorUtils.h>
#include <torch/library.h>
#include <c10/util/irange.h>
#include <c10/util/strides.h>

namespace {
  void functionalizeFallback(const c10::OperatorHandle& op, c10::DispatchKeySet dispatchKeySet, torch::jit::Stack* stack) {
    const auto& schema = op.schema();
    TORCH_CHECK(
      !schema.hasAnyAliasInfo(),
      "Found a custom (non-ATen) operator that either mutates or its inputs: ",
      op.operator_name().name, ".", op.operator_name().overload_name,
      ". Getting these operators to work with functionalization requires some extra work",
      ". For mutable ops you need to register a corresponding out-of-place variant of the op,",
      " and you also need to register a Functionalization kernel that performs some boilerplate,",
      " telling functionalization to map from the mutable op to the out-of-place op",
      ". See a more complete example of how to do this at ",
      "https://gist.github.com/bdhirsh/7dadbf6296f8f7d1abcf4c482f438aaa.",
      " Please file a GitHub issue if you run into any problems.");
    const auto num_arguments = schema.arguments().size();
    const auto arguments_begin = stack->size() - num_arguments;
    auto arguments = torch::jit::last(stack, num_arguments);

    auto any_functional_inputs = false;
    auto any_tensor_inputs = false;
    for (uint64_t idx = 0; idx < num_arguments; ++idx) {
      const auto& ivalue = arguments[idx];
      if (ivalue.isTensor()) {
        any_tensor_inputs = true;
        const auto& t = ivalue.toTensor();
        if (t.defined() && at::functionalization::impl::isFunctionalTensor(t)) {
          any_functional_inputs = true;
          at::functionalization::impl::sync(t);
          auto t_new = c10::IValue(at::functionalization::impl::from_functional_tensor(t));
          (*stack)[arguments_begin + idx] = t_new;
        }
      } else if (ivalue.isTensorList()) {
        any_tensor_inputs = true;
        auto tensors = ivalue.toTensorList();
        if (at::functionalization::impl::isFunctionalTensor(tensors)) {
          any_functional_inputs = true;
          at::functionalization::impl::sync(tensors);
          auto t_new = c10::IValue(at::functionalization::impl::from_functional_tensor(tensors));
          (*stack)[arguments_begin + idx] = t_new;
        }
      } else if (ivalue.isOptionalTensorList()) {
        any_tensor_inputs = true;
        auto opt_tensors = ivalue.toOptionalTensorList();
        if (at::functionalization::impl::isFunctionalTensor(opt_tensors)) {
          any_functional_inputs = true;
          at::functionalization::impl::sync(opt_tensors);
          auto t_new = c10::IValue(at::functionalization::impl::from_functional_tensor(opt_tensors));
          (*stack)[arguments_begin + idx] = t_new;
        }
      }
    }
    // we should wrap the output if any inputs were wrapped,
    // OR if we're hitting a factory function (with no tensor inputs)
    auto should_wrap_outputs = !any_tensor_inputs || any_functional_inputs;
    {
      at::AutoDispatchSkipFunctionalize guard;
      op.callBoxed(stack);
    }
    const auto num_returns = schema.returns().size();
    const auto returns_begin = stack->size() - num_returns;
    auto returns = torch::jit::last(stack, num_returns);

    for (const auto idx : c10::irange(num_returns)) {
      const auto& ivalue = returns[idx];
      if (ivalue.isTensor() && should_wrap_outputs) {
        const auto& t = ivalue.toTensor();
        if (!t.defined()) continue;
        auto t_new = c10::IValue(at::functionalization::impl::to_functional_tensor(t));
        (*stack)[returns_begin + idx] = t_new;
      } else if (ivalue.isTensorList() && should_wrap_outputs) {
        auto tensors = ivalue.toTensorList();
        auto t_new = c10::IValue(at::functionalization::impl::to_functional_tensor(tensors));
        (*stack)[returns_begin + idx] = t_new;
      } else if (ivalue.isOptionalTensorList() && should_wrap_outputs) {
        auto opt_tensors = ivalue.toOptionalTensorList();
        auto t_new = c10::IValue(at::functionalization::impl::to_functional_tensor(opt_tensors));
        (*stack)[returns_begin + idx] = t_new;
      }
    }
  }
}

TORCH_LIBRARY_IMPL(_, Functionalize, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&functionalizeFallback>());
}
